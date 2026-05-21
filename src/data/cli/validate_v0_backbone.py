"""Stage 1.9 validation: hybrid SSM + attention backbone integrity.

Six focused tests, per the Stage 1.9 plan:

  1. test_block_io_contract_preserved        — SSMBlock and AttnBlock keep (B,T,d_model)
  2. test_ssm_pad_isolation                  — pad positions don't contaminate the SSM state
  3. test_block_position_pattern             — _ssm_positions matches the planned table
  4. test_hybrid_jepa_overfits_single_batch  — gradients flow through SSM blocks; MSE drops
  5. test_hybrid_training_run_50_steps       — end-to-end CLI + checkpoint roundtrip
  6. test_param_count_in_budget              — hybrid stays inside the v0 param envelope

The existing 5 v0 tests (validate_v0_training) continue to gate the
transformer path; the new tests below exercise the hybrid path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from training.bwm_v0.blocks import (
    AttnBlock,
    HybridBackbone,
    SSMBlock,
    _ssm_positions,
)
from training.bwm_v0.model import JEPABackbone, JEPAModel
from training.bwm_v0.train import train_loop


def _hash_state_dict(sd: dict) -> str:
    import hashlib
    h = hashlib.sha1()
    for k in sorted(sd.keys()):
        v = sd[k]
        if isinstance(v, torch.Tensor):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# ---------- test 1 ---------------------------------------------------------


def test_block_io_contract_preserved() -> None:
    print("\n--- test 1: block I/O contracts preserved ---")
    B, T, D = 4, 32, 64
    x = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, :10] = 0  # left-pad sample 0

    ssm = SSMBlock(d_model=D)
    out_ssm = ssm(x, mask)
    assert out_ssm.shape == (B, T, D), f"SSMBlock out {out_ssm.shape}"
    assert bool(torch.isfinite(out_ssm).all()), "SSMBlock produced non-finite values"

    attn = AttnBlock(d_model=D, n_heads=4)
    out_attn = attn(x, mask)
    assert out_attn.shape == (B, T, D), f"AttnBlock out {out_attn.shape}"
    assert bool(torch.isfinite(out_attn).all()), "AttnBlock produced non-finite values"

    print(f"  [OK] SSMBlock {tuple(out_ssm.shape)}, AttnBlock {tuple(out_attn.shape)}, all finite")


# ---------- test 2 ---------------------------------------------------------


def test_ssm_pad_isolation() -> None:
    """SSM is causal left-to-right. Pad positions (mask=0) must contribute
    nothing to the output at real positions. We feed two batches with
    identical real-data tails but DIFFERENT junk in their pad regions, and
    check that the SSM block's output at real positions is bit-identical.

    The block has two pad-isolation mechanisms working together:
      (a) x_branch is zeroed at pad positions BEFORE the causal conv, so
          the conv can't mix pad values into adjacent real positions
      (b) B_bar is zeroed at pad positions inside the SSM scan, so even
          if conv-mixed inputs survived, pad positions contribute nothing
          to the recurrence state

    Together, mechanism (a) cuts off conv contamination; mechanism (b) is
    belt-and-suspenders for the scan itself. The integration test below
    uses block.forward() (which applies both) and asserts perfect isolation.
    The residual connection from x at pad positions is excluded — it only
    affects pad output positions, which we slice away.
    """
    print("\n--- test 2: SSM pad isolation ---")
    torch.manual_seed(123)
    D = 32
    real_tail = torch.randn(1, 16, D)
    pad_a = torch.zeros(1, 16, D)
    x_a = torch.cat([pad_a, real_tail], dim=1)
    pad_b = torch.randn(1, 16, D) * 10.0
    x_b = torch.cat([pad_b, real_tail], dim=1)

    mask = torch.zeros(1, 32, dtype=torch.long)
    mask[0, 16:] = 1  # left-pad: first 16 pad, last 16 real

    block = SSMBlock(d_model=D)
    block.eval()  # disable dropout
    with torch.no_grad():
        y_a = block(x_a, mask)
        y_b = block(x_b, mask)

    # Block forward returns x + dropout(y). At real positions the residual
    # x term is identical between A and B (since real_tail is shared). The
    # `y` path is what we're testing — and it should be identical too.
    tail_diff = (y_a[:, 16:] - y_b[:, 16:]).abs().max().item()
    print(f"  max abs diff at real positions [16:32]: {tail_diff:.6e}")
    # Strict zero tolerance: pad isolation should be exact (bit-identical
    # because both upstream paths produce identical tensors at real positions).
    assert tail_diff < 1e-6, (
        f"SSM pad isolation broken: real-tail outputs differ by {tail_diff:.3e} "
        f"between samples with identical real data but different pad junk"
    )
    print("  [OK] SSM block isolates real positions from pad-region variation (exact)")


# ---------- test 3 ---------------------------------------------------------


def test_block_position_pattern() -> None:
    print("\n--- test 3: block position-picking pattern ---")
    cases = [
        (4, 0.0,  set()),
        (4, 0.25, {0}),
        (4, 0.5,  {0, 2}),       # alternating override
        (4, 0.75, {0, 1, 2}),
        (4, 1.0,  {0, 1, 2, 3}),
    ]
    for n, r, expected in cases:
        got = _ssm_positions(n, r)
        assert got == expected, f"n={n}, ratio={r}: got {got}, expected {expected}"
        print(f"  n={n}, ratio={r}: positions={sorted(got)}")

    # HybridBackbone surfaces n_ssm + n_attn correctly
    hb = HybridBackbone(d_model=64, n_layers=4, ssm_ratio=0.75)
    assert hb.n_ssm == 3, f"n_ssm={hb.n_ssm} expected 3"
    assert hb.n_attn == 1, f"n_attn={hb.n_attn} expected 1"
    assert hb.block_types == ["ssm", "ssm", "ssm", "attn"]
    print(f"  [OK] HybridBackbone(n=4, ratio=0.75): types={hb.block_types}")


# ---------- test 4 ---------------------------------------------------------


def test_hybrid_jepa_overfits_single_batch() -> None:
    print("\n--- test 4: hybrid JEPA overfits a fixed batch ---")
    os.environ.setdefault("BWM_DATA_ROOT", ".data")

    summary = train_loop(
        steps=200,
        batch_size=4,
        d_model=64,
        n_layers=4,
        seed=42,
        warmup_steps=10,
        out_dir="/tmp/v0_hybrid_overfit",
        overfit_single_batch=True,
        ckpt_every=1000,
        backbone_type="hybrid",
        ssm_ratio=0.75,
    )
    rows = [json.loads(l) for l in open(summary["log_path"])]
    assert len(rows) >= 50
    first_mse = rows[5]["mse"]
    last_mse = rows[-1]["mse"]
    print(f"  step-5 MSE = {first_mse:.4f};  step-{rows[-1]['step']} MSE = {last_mse:.4f}")
    assert all(np.isfinite(r["total"]) for r in rows), "non-finite loss appeared"
    assert last_mse < first_mse * 0.5, (
        f"hybrid single-batch overfit failed: MSE only dropped {first_mse:.4f} → {last_mse:.4f}"
    )
    print(f"  [OK] MSE reduced {first_mse:.4f} → {last_mse:.4f} ({(1-last_mse/first_mse)*100:.0f}% drop)")


# ---------- test 5 ---------------------------------------------------------


def test_hybrid_training_run_50_steps() -> None:
    print("\n--- test 5: hybrid 50-step run + checkpoint roundtrip ---")
    os.environ.setdefault("BWM_DATA_ROOT", ".data")
    out_dir = Path("/tmp/v0_hybrid_50")
    if out_dir.exists():
        for f in out_dir.glob("*"):
            f.unlink()

    summary = train_loop(
        steps=50,
        batch_size=4,
        d_model=64,
        n_layers=4,
        seed=42,
        warmup_steps=10,
        out_dir=str(out_dir),
        ckpt_every=50,
        backbone_type="hybrid",
        ssm_ratio=0.75,
    )
    assert summary["steps_completed"] == 50

    rows = [json.loads(l) for l in open(summary["log_path"])]
    for r in rows:
        for k in ("total", "mse", "var", "cov"):
            assert np.isfinite(r[k]), f"non-finite {k} at step {r['step']}: {r[k]}"
        assert r["mse"] >= 0 and r["var"] >= 0 and r["cov"] >= 0
    print(f"  [OK] all {len(rows)} log rows finite, components non-negative")

    ckpt_file = out_dir / "step-50.pt"
    assert ckpt_file.exists()
    ckpt = torch.load(ckpt_file, weights_only=False)
    assert ckpt["config"]["backbone_type"] == "hybrid"
    assert ckpt["config"]["ssm_ratio"] == 0.75
    online_hash_saved = _hash_state_dict(ckpt["online_state_dict"])

    cfg = ckpt["config"]
    fresh = JEPAModel(
        d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        ff_dim=cfg["ff_dim"], dropout=cfg["dropout"],
        max_history_q=cfg["max_history_q"],
        n_horizons=len(cfg["horizons"]),
        predictor_hidden=cfg["predictor_hidden"],
        backbone_type=cfg["backbone_type"],
        ssm_ratio=cfg["ssm_ratio"],
        d_state=cfg["d_state"], d_conv=cfg["d_conv"],
    )
    fresh.online.load_state_dict(ckpt["online_state_dict"])
    fresh.target.load_state_dict(ckpt["target_state_dict"])
    fresh.predictor.load_state_dict(ckpt["predictor_state_dict"])
    online_hash_reloaded = _hash_state_dict(fresh.online.state_dict())
    assert online_hash_saved == online_hash_reloaded
    print(f"  [OK] hybrid checkpoint roundtrip byte-identical (SHA1={online_hash_saved[:12]}...)")


# ---------- test 6 ---------------------------------------------------------


def test_param_count_in_budget() -> None:
    print("\n--- test 6: hybrid param count in budget ---")
    # Pure transformer baseline (Stage 1.8 defaults)
    xfmr = JEPAModel(d_model=64, n_layers=2, n_heads=4, backbone_type="transformer")
    n_xfmr = sum(p.numel() for p in xfmr.parameters() if p.requires_grad)
    print(f"  transformer(d=64, n=2): {n_xfmr:,} trainable params")

    # Hybrid (Stage 1.9 defaults)
    hyb = JEPAModel(d_model=64, n_layers=4, n_heads=4, backbone_type="hybrid", ssm_ratio=0.75)
    n_hyb = sum(p.numel() for p in hyb.parameters() if p.requires_grad)
    print(f"  hybrid(d=64, n=4, ratio=0.75): {n_hyb:,} trainable params")

    assert 200_000 <= n_hyb <= 400_000, (
        f"hybrid trainable param count {n_hyb:,} outside expected [200K, 400K] budget"
    )
    print(f"  [OK] hybrid params {n_hyb:,} within [200K, 400K] budget")


# ---------- driver ---------------------------------------------------------


def main() -> None:
    print("=== Stage 1.9 validation: hybrid SSM + attention backbone ===")
    test_block_io_contract_preserved()
    test_ssm_pad_isolation()
    test_block_position_pattern()
    test_hybrid_jepa_overfits_single_batch()
    test_hybrid_training_run_50_steps()
    test_param_count_in_budget()
    print("\n=== Stage 1.9 validation: PASS ===")


if __name__ == "__main__":
    main()
