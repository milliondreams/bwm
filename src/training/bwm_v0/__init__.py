"""BWM v0 training pipeline (Stage 1.8 / Definition A).

The smallest viable JEPA training loop that exercises every layer of the
training pipeline end-to-end. Goal is integration correctness, not capacity.

Deliberate v0 simplifications versus full Phase B per spec § 4.1 / § 6.3:
  # v0: pure transformer encoder (Phase B uses hybrid Mamba-3 + attention; ADR-1 fallback)
  # v0: financials modality only (Phase B fuses all 10)
  # v0: 30 hand-curated XBRL concepts (Phase B uses all)
  # v0: ~175K params (Phase B sweeps 150M/350M/700M)
  # v0: ≤32 quarter sequences, no graph context
  # v0: horizons h ∈ {1, 4}Q (Phase B uses {1, 2, 4, 8, 12})
  # v0: no action conditioning (passive forecasting)
  # v0: CPU, single-process

Each simplification is independently liftable to Phase B without touching others.
"""
