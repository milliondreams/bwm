# BWM — Business World Model

A learned representation of how business entities evolve over time. Given a
company's current state, BWM predicts likely future states, quantifies
uncertainty, supports counterfactual reasoning about actions, and produces
auditable explanations for every prediction.

Full design spec: [`initial-docs/business-world-model-spec-v3 (1).md`](initial-docs/business-world-model-spec-v3%20%281%29.md).

## Status

Phase A (data foundation) is code-complete; Phase B (model pretraining at
scale) is in v0 / integration-test form.

| Layer | State |
|---|---|
| 10-modality data ingest (SEC EDGAR, FRED, yfinance, GDELT, USPTO, BLS, ...) | code complete |
| Point-in-time engine + EntitySnapshot (FR-2 contract) | proven against real GE restatement |
| v3 PIT field set (`belief`, `perspective`, `policy_tags`, `source_certainty`, ...) | landed |
| Data-quality CVR gate (bwm.accounting + bwm.regulation) | 0.00% violations |
| Training-time PIT awareness (`PITDataset`, `EntitySnapshot`) | adversarial-leak tested |
| JEPA training loop v0 (Stage 1.8 — pure transformer baseline) | 5 integration tests |
| Hybrid SSM + attention backbone v0 (Stage 1.9 — ADR-1) | 6 integration tests |
| Phase B at-scale training (GPU, full corpus, sweep) | not yet — see roadmap |

**19 of 19 validation suites passing** at the time of this snapshot.

## Layout

```
src/
  data/
    schemas/       Pydantic models — 10 modality records + PIT envelope
    sources/       Connectors (SEC EDGAR, yfinance, FRED, GDELT, USPTO, BLS)
    pit/           PIT engine + EntitySnapshot
    training/      PITDataset — PIT-safe training tuples
    validation/    bwm.accounting + bwm.regulation rule engine
    cli/           Ingest + validation CLIs (19 validate_*.py suites)
  training/
    bwm_v0/        Definition A: JEPA training loop, transformer or hybrid backbone
  aml/             Azure ML pipeline scaffolding
initial-docs/      Spec (v1 + v3)
```

## Setup

```bash
uv sync                                # creates .venv, installs deps from pyproject.toml
cp .env.example .env                   # fill in Azure subscription / RG / workspace
set -a && source .env && set +a
```

## Run the validation suites

```bash
# Foundation
uv run python -m data.cli.smoke_stage1
uv run python -m data.cli.smoke_stage1_5

# PIT engine + restatement
uv run python -m data.cli.validate_restatement
uv run python -m data.cli.validate_pit_training

# Per-modality (EDGAR financials, Form 4, filings text, supply chain;
# market, macro, news, patents, earnings calls, hiring)
uv run python -m data.cli.validate_xbrl_contexts
uv run python -m data.cli.validate_form4
uv run python -m data.cli.validate_filing_text
uv run python -m data.cli.validate_stage6
uv run python -m data.cli.validate_market
uv run python -m data.cli.validate_macro
uv run python -m data.cli.validate_patents
uv run python -m data.cli.validate_news
uv run python -m data.cli.validate_earnings_calls
uv run python -m data.cli.validate_hiring

# Data quality + v3 conformance
uv run python -m data.cli.validate_v3_fields
uv run python -m data.cli.validate_constraints

# v0 training (JEPA + hybrid backbone)
uv run python -m data.cli.validate_v0_training
uv run python -m data.cli.validate_v0_backbone

# Phase A coverage report
uv run python -m data.cli.validate_stage5
```

## Roadmap (next steps)

1. **Stage 5 full-corpus EDGAR backfill** — `ingest_edgar_full --all` on the
   full ~10K US public company registry (~47 wall hours at 9 req/s)
2. **Probe harness** for §9.2 acceptance gates (linear probes on the v0
   encoder for regime classification, margin direction, layoff event)
3. **Multimodal features** — extend `bwm_v0/features.py` to fuse market,
   macro, filings_text, etc. (currently financials only)
4. **GPU + scale** — port the training loop to Azure ML compute when GPU
   quota lands; swap pure-PyTorch SSM scan for `mamba-ssm` CUDA kernels
5. **Stage 6 live LLM IE** — supply-chain edge extraction from 10-K narrative
   via Azure OpenAI (the harness is wired; needs operational launch)

## License

TBD.
