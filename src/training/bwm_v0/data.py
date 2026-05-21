"""torch.utils.data.Dataset wrapping PITDataset for the v0 training loop.

PITDataset already enforces the PIT contract: each yielded
PITTrainingExample carries an input_snapshot bounded at as_of=t and a
target_snapshot bounded at as_of=t+h. This wrapper tensorizes both via
snapshot_to_tensor and produces a flat dict for collation.

Skip-policy:
  - Examples with empty input_snapshot.financials (no signal to encode)
    are skipped at materialization.
  - Examples with empty target_snapshot are KEPT — the model learns
    "nothing knowable yet at t+h" via a zero target tensor.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from data.training.dataset import PITDataset
from training.bwm_v0.features import snapshot_to_tensor


# v0 horizons. Phase B uses {1, 2, 4, 8, 12}.
HORIZONS_V0: tuple[int, ...] = (1, 4)


def horizon_to_index(h: int) -> int:
    """Map a horizon (in quarters) to the predictor's embedding index."""
    return HORIZONS_V0.index(h)


class BWMTrainingDataset(Dataset):
    """Concrete, indexable list of (input, target, horizon) tensors.

    Materializes PITDataset's generator into a list at construction time.
    PITDataset is small (15 entities × ~28 quarters × 2 horizons ≈ 840
    examples before filtering); we keep the whole thing in RAM.
    """

    def __init__(
        self,
        pit_dataset: PITDataset,
        max_history_q: int = 32,
        verbose: bool = False,
    ) -> None:
        self.max_history_q = max_history_q
        self._examples: list[dict] = []
        n_total = 0
        n_skipped_empty_input = 0
        for ex in pit_dataset:
            n_total += 1
            inp = snapshot_to_tensor(ex.input_snapshot, max_history_q)
            if inp["seq_len"] == 0:
                n_skipped_empty_input += 1
                continue
            tgt = snapshot_to_tensor(ex.target_snapshot, max_history_q)
            self._examples.append({
                "input_x": inp["x"],
                "input_mask": inp["mask"],
                "input_seq_len": inp["seq_len"],
                "target_x": tgt["x"],
                "target_mask": tgt["mask"],
                "target_seq_len": tgt["seq_len"],
                "horizon_q": int(ex.horizon_quarters),
                "entity_id": ex.entity_id,
                "t": ex.t.isoformat(),
            })
        if verbose:
            print(
                f"BWMTrainingDataset: materialized {len(self._examples)} of "
                f"{n_total} PIT examples ({n_skipped_empty_input} skipped — empty input)"
            )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self._examples[idx]
        return {
            "input_x": torch.from_numpy(ex["input_x"]),
            "input_mask": torch.from_numpy(ex["input_mask"]),
            "target_x": torch.from_numpy(ex["target_x"]),
            "target_mask": torch.from_numpy(ex["target_mask"]),
            "horizon_q": torch.tensor(ex["horizon_q"], dtype=torch.long),
            # Pass-through string fields for debugging; collated as lists.
            "entity_id": ex["entity_id"],
            "t": ex["t"],
        }
