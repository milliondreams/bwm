"""v0 JEPA training CLI.

Usage:
    uv run python -m training.bwm_v0.train --steps 200 --batch-size 4 --d-model 64 \
        --out checkpoints/v0/smoke

The loop is deliberately minimal (see Stage 1.8 plan): pure transformer
backbone, financials-only, 30 concepts, 32-quarter history, horizons
{1, 4}Q, AdamW + cosine schedule, CPU. Phase B replaces each piece in
isolation without changing the loop structure.

Logs JSONL per step to <out>/train.log.jsonl. Checkpoints to <out>/step-<N>.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.entity.registry import EntityRegistry
from data.pit.engine import PITEngine
from data.storage import get_storage
from data.training.dataset import PITDataset, sample_training_times
from training.bwm_v0.data import BWMTrainingDataset, HORIZONS_V0, horizon_to_index
from training.bwm_v0.ema import ema_update
from training.bwm_v0.loss import jepa_loss
from training.bwm_v0.model import JEPAModel


# Default entity set for v0 — the CIKs we have canonical data for.
DEFAULT_CIKS: tuple[str, ...] = (
    "0000320193",  # Apple
    "0000789019",  # Microsoft
    "0001018724",  # Amazon
    "0001045810",  # NVIDIA
    "0001318605",  # Tesla
    "0001067983",  # Berkshire
    "0001652044",  # Alphabet
    "0001326801",  # Meta
    "0000104169",  # Walmart
    "0000019617",  # JPMorgan
    "0000040545",  # GE
)


def set_seeds(seed: int) -> None:
    """Pin RNG state across torch / numpy / Python random for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cosine_warmup_schedule(step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1) -> float:
    """Linear warmup then cosine decay to min_lr_ratio. Returns scale factor."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_dataset(
    ciks: list[str],
    start: date,
    end: date,
    horizons: tuple[int, ...] = HORIZONS_V0,
    max_history_q: int = 32,
    shuffle: bool = True,
    seed: int = 42,
    verbose: bool = False,
) -> BWMTrainingDataset:
    """Build the v0 training dataset from canonical financials."""
    storage = get_storage()
    pit = PITEngine(storage)
    entity_ids = [EntityRegistry.entity_id_from_cik(c) for c in ciks]
    times = sample_training_times(start, end)
    pit_ds = PITDataset(
        pit=pit,
        entity_ids=entity_ids,
        times=times,
        horizons=list(horizons),
        shuffle=shuffle,
        seed=seed,
    )
    return BWMTrainingDataset(pit_ds, max_history_q=max_history_q, verbose=verbose)


def train_loop(
    *,
    steps: int = 200,
    batch_size: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    ff_dim: int = 256,
    dropout: float = 0.1,
    predictor_hidden: int = 128,
    backbone_type: str = "transformer",  # Stage 1.9: "transformer" | "hybrid"
    ssm_ratio: float = 0.75,              # Stage 1.9: used iff backbone_type=="hybrid"
    d_state: int = 16,
    d_conv: int = 4,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    warmup_steps: int = 20,
    grad_clip_norm: float = 1.0,
    seed: int = 42,
    tau: float = 0.996,
    gamma: float = 1.0,
    lambda_var: float = 1.0,
    lambda_cov: float = 0.05,
    max_history_q: int = 32,
    log_every: int = 1,
    ckpt_every: int = 50,
    overfit_single_batch: bool = False,
    ciks: Optional[list[str]] = None,
    start_year: int = 2018,
    end_year: int = 2024,
    out_dir: str = "checkpoints/v0/smoke",
) -> dict:
    """Run a JEPA training loop. Returns a summary dict with the final state."""
    set_seeds(seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_path = out_path / "train.log.jsonl"

    config = {
        "steps": steps, "batch_size": batch_size, "d_model": d_model,
        "n_layers": n_layers, "n_heads": n_heads, "ff_dim": ff_dim,
        "dropout": dropout, "predictor_hidden": predictor_hidden,
        "backbone_type": backbone_type, "ssm_ratio": ssm_ratio,
        "d_state": d_state, "d_conv": d_conv,
        "lr": lr, "weight_decay": weight_decay, "warmup_steps": warmup_steps,
        "grad_clip_norm": grad_clip_norm, "seed": seed, "tau": tau,
        "gamma": gamma, "lambda_var": lambda_var, "lambda_cov": lambda_cov,
        "max_history_q": max_history_q, "horizons": list(HORIZONS_V0),
        "n_concepts": 30,
        "ciks": list(ciks or DEFAULT_CIKS),
        "start_year": start_year, "end_year": end_year,
        "overfit_single_batch": overfit_single_batch,
    }

    print(
        f"=== bwm_v0 training: {steps} steps, batch={batch_size}, "
        f"d_model={d_model}, backbone={backbone_type}"
        f"{f' (ssm_ratio={ssm_ratio})' if backbone_type == 'hybrid' else ''} ==="
    )
    ds = build_dataset(
        ciks=config["ciks"],
        start=date(start_year, 1, 1),
        end=date(end_year, 12, 31),
        max_history_q=max_history_q,
        shuffle=True,
        seed=seed,
        verbose=True,
    )
    if len(ds) == 0:
        raise SystemExit("training dataset is empty — check canonical financials exist")

    generator = torch.Generator().manual_seed(seed)
    if overfit_single_batch:
        # Reduce to one fixed batch by indexing the first batch_size examples
        # and repeating the same DataLoader output across steps.
        indices = list(range(min(batch_size, len(ds))))
        fixed_batch_examples = [ds[i] for i in indices]
        fixed_batch = _collate(fixed_batch_examples)
        loader = None
    else:
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
            collate_fn=_collate,
            drop_last=True,
        )

    model = JEPAModel(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, ff_dim=ff_dim,
        dropout=dropout, max_history_q=max_history_q,
        n_horizons=len(HORIZONS_V0), predictor_hidden=predictor_hidden,
        backbone_type=backbone_type, ssm_ratio=ssm_ratio,
        d_state=d_state, d_conv=d_conv,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95),
    )

    log_f = open(log_path, "w")
    t0 = time.monotonic()
    step = 0
    final_loss = None
    final_components = None
    try:
        iterator = _infinite_loader(loader, fixed_batch if overfit_single_batch else None)
        for step in range(steps):
            batch = next(iterator)
            input_x = batch["input_x"]
            input_mask = batch["input_mask"]
            target_x = batch["target_x"]
            target_mask = batch["target_mask"]
            horizon_q = batch["horizon_q"]
            horizon_idx = torch.tensor(
                [horizon_to_index(int(h)) for h in horizon_q.tolist()],
                dtype=torch.long,
            )

            s_t = model.online(input_x, input_mask)
            s_target = model.target_encode(target_x, target_mask)
            pred = model.predictor(s_t, horizon_idx)

            loss, comps = jepa_loss(
                pred, s_target, s_t,
                gamma=gamma, lambda_var=lambda_var, lambda_cov=lambda_cov,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at step {step}: total={loss.item()} "
                    f"mse={comps['mse'].item()} var={comps['var'].item()} cov={comps['cov'].item()}"
                )

            # LR schedule
            scale = cosine_warmup_schedule(step, warmup_steps, steps)
            for g in optimizer.param_groups:
                g["lr"] = lr * scale

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip_norm
            )
            optimizer.step()
            ema_update(model.target, model.online, tau=tau)

            if step % log_every == 0 or step == steps - 1:
                row = {
                    "step": step,
                    "lr": lr * scale,
                    "mse": float(comps["mse"].item()),
                    "var": float(comps["var"].item()),
                    "cov": float(comps["cov"].item()),
                    "total": float(comps["total"].item()),
                    "elapsed_s": round(time.monotonic() - t0, 3),
                }
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()
                if step % max(1, log_every * 10) == 0 or step == steps - 1:
                    print(
                        f"  step {step:>4}  total={row['total']:.4f}  "
                        f"mse={row['mse']:.4f}  var={row['var']:.4f}  cov={row['cov']:.4f}  "
                        f"lr={row['lr']:.2e}"
                    )

            if (step + 1) % ckpt_every == 0 or step == steps - 1:
                ckpt_file = out_path / f"step-{step + 1}.pt"
                torch.save(
                    {
                        "step": step + 1,
                        "config": config,
                        "online_state_dict": model.online.state_dict(),
                        "target_state_dict": model.target.state_dict(),
                        "predictor_state_dict": model.predictor.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    ckpt_file,
                )

            final_loss = float(loss.item())
            final_components = {k: float(v.item()) for k, v in comps.items()}
    finally:
        log_f.close()

    elapsed = time.monotonic() - t0
    summary = {
        "steps_completed": step + 1 if final_loss is not None else 0,
        "final_loss": final_loss,
        "final_components": final_components,
        "elapsed_s": round(elapsed, 2),
        "out_dir": str(out_path),
        "log_path": str(log_path),
        "config": config,
    }
    print(
        f"\n=== complete: {summary['steps_completed']} steps in {summary['elapsed_s']}s, "
        f"final total={final_loss:.4f} ==="
    )
    return summary


def _collate(examples: list[dict]) -> dict:
    """Stack tensor fields; pass through string fields as lists."""
    return {
        "input_x": torch.stack([e["input_x"] for e in examples]),
        "input_mask": torch.stack([e["input_mask"] for e in examples]),
        "target_x": torch.stack([e["target_x"] for e in examples]),
        "target_mask": torch.stack([e["target_mask"] for e in examples]),
        "horizon_q": torch.stack([e["horizon_q"] for e in examples]),
        "entity_id": [e["entity_id"] for e in examples],
        "t": [e["t"] for e in examples],
    }


def _infinite_loader(loader: Optional[DataLoader], fixed_batch: Optional[dict]):
    """Yield batches forever; either by cycling a DataLoader or yielding a fixed batch."""
    if fixed_batch is not None:
        while True:
            yield fixed_batch
    assert loader is not None
    while True:
        for batch in loader:
            yield batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument(
        "--n-layers", type=int, default=None,
        help="Encoder block count. Default: 2 for backbone=transformer, 4 for hybrid.",
    )
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-history-q", type=int, default=32)
    ap.add_argument("--data-root", default=".data")
    ap.add_argument("--out", default="checkpoints/v0/smoke")
    ap.add_argument("--overfit-single-batch", action="store_true",
                    help="Train repeatedly on the first batch — sanity test that loss decreases")
    ap.add_argument("--ckpt-every", type=int, default=50)
    # Stage 1.9 hybrid-backbone flags
    ap.add_argument(
        "--backbone", choices=("transformer", "hybrid"), default="transformer",
        help="Encoder family. 'transformer' (default, Stage 1.8 path) or "
             "'hybrid' (Stage 1.9: SSM + attention per ADR-1).",
    )
    ap.add_argument(
        "--ssm-ratio", type=float, default=0.75,
        help="Fraction of blocks that are SSM in the hybrid backbone. "
             "0.0 = all attention; 0.5 = 1:1; 0.75 = 3:1 (ADR-1 prior); 1.0 = pure SSM.",
    )
    args = ap.parse_args()

    # Default n_layers per backbone — 2 for transformer (Stage 1.8 compat),
    # 4 for hybrid (so 3:1 ratio is exact).
    n_layers = args.n_layers
    if n_layers is None:
        n_layers = 4 if args.backbone == "hybrid" else 2

    os.environ.setdefault("BWM_DATA_ROOT", args.data_root)
    train_loop(
        steps=args.steps,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_layers=n_layers,
        n_heads=args.n_heads,
        lr=args.lr,
        seed=args.seed,
        max_history_q=args.max_history_q,
        out_dir=args.out,
        overfit_single_batch=args.overfit_single_batch,
        ckpt_every=args.ckpt_every,
        backbone_type=args.backbone,
        ssm_ratio=args.ssm_ratio,
    )


if __name__ == "__main__":
    main()
