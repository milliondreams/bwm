# Business World Model (BWM): Technical Specification

**Architecture:** SSM backbone + JEPA latent prediction + Diffusion uncertainty head + Neuro-symbolic constraint and explanation layer

**Status:** Design specification
**Audience:** Engineering, research, product, and executive sponsors

---

## 1. Executive Summary

The Business World Model (BWM) is a learned representation of how business entities evolve over time. Given a company's current state, the BWM predicts likely future states, quantifies uncertainty, allows counterfactual reasoning about actions, and produces auditable explanations for every prediction.

It is designed to be the underlying *intelligence layer* for products in three categories:

- **Early-warning systems** (credit risk, supply-chain disruption, fraud, audit alerts)
- **Scenario simulation** (M&A modeling, strategic planning, stress testing)
- **Decision support** (deal screening, action recommendation, portfolio monitoring)

The architecture is a layered hybrid system. No single component is novel; the design contribution is the integration. Each layer addresses a specific enterprise-buy requirement that the others cannot satisfy alone:

| Layer | Component | Job |
|---|---|---|
| 1 | Hybrid SSM + attention backbone (Mamba-3 + interleaved self-attention; ADR-1) | Encode long entity histories with precise event recall |
| 2 | JEPA latent prediction head | Learn abstract regime-level state |
| 3 | Diffusion / flow head | Generate distributions over futures |
| 4 | Neuro-symbolic layer | Enforce constraints, generate explanations |
| 5 | Planning interface | Rank actions and produce decision-ready output |

---

## 2. Goals and Non-Goals

### 2.1 Goals

- Predict business entity state evolution over horizons of 1–12 quarters
- Produce calibrated probability distributions over future states
- Support counterfactual reasoning under candidate actions
- Generate symbolic explanations that pass regulator / auditor scrutiny
- Scale to ~50,000 entities with 20+ years of multimodal history
- Support both public (foundation) and private (customer) data integration
- Be evaluated against historical crisis periods as held-out tests

### 2.2 Non-Goals

- **Price prediction.** The system predicts business state, not market prices. Trading applications are explicitly out of scope.
- **Causal claims without experiments.** The system surfaces hypotheses and analogs; it does not assert causation.
- **Single-company modeling without peers.** The system requires a peer/graph context to function; isolated single-entity use is unsupported.
- **Real-time millisecond inference.** Quarterly cadence with intra-quarter event updates is the design point. Sub-second latency is not required.
- **Frontier-LLM-style generality.** This is a domain model. Text inputs are encoded via pretrained LMs but the system does not produce general-purpose text.

---

## 3. Requirements

### 3.1 Functional Requirements

**FR-1: State Encoding**
The system shall encode any entity at any point in time as a continuous latent vector representing its business state, computed from all available historical data up to that point.

**FR-2: Point-in-Time Integrity**
All training and inference shall respect point-in-time data availability. The system shall never use information that was not knowable on the prediction date (no peeking at restated financials).

**FR-3: Horizon-Conditioned Prediction**
The system shall predict latent state at user-specified horizons of {1, 2, 4, 8, 12} quarters. Predictions at unsupported horizons shall be interpolated explicitly, not silently extrapolated.

**FR-4: Action Conditioning**
The system shall accept structured actions (acquisition, layoff, pricing change, capital raise, market entry, etc.) and produce action-conditional state predictions.

**FR-5: Distribution over Futures**
The system shall produce, for any (state, action, horizon) tuple, a sample of N ≥ 500 plausible futures from which P10/P50/P90 trajectories on downstream variables are derived.

**FR-6: Constraint Compliance**
The system shall validate all sampled futures against a set of hard constraints (accounting identities, sign constraints, logical impossibilities) and reject violators before they reach the user.

**FR-7: Explanation Generation**
For every prediction surfaced to a user, the system shall produce a symbolic trace identifying the top-K rules and graph edges that contributed to the prediction.

**FR-8: Analog Retrieval**
For every prediction, the system shall retrieve the K nearest historical analogs by latent-space distance and surface their actual outcomes.

**FR-9: Graph Awareness**
Predictions for entity A shall be influenced by the state of A's graph neighbors (suppliers, customers, competitors) through learned message passing or attention.

**FR-10: Update Cadence**
The system shall support both batch quarterly retraining and event-driven incremental updates (new filings, M&A announcements, news shocks).

### 3.2 Non-Functional Requirements

**NFR-1: Reproducibility**
All training runs shall be reproducible from a versioned data snapshot, code commit, and config. A prediction made on date D shall be recoverable indefinitely.

**NFR-2: Auditability**
All inference shall log inputs, intermediate activations sufficient for explanation, the explanation trace, and outputs. Logs shall be retained per regulatory requirements (default: 7 years).

**NFR-3: Latency**
- Single-entity, single-horizon prediction: < 1 second
- Single-entity, distribution over 1000 samples: < 30 seconds
- Full-universe batch (50K entities, 5 horizons): < 8 hours

**NFR-4: Throughput**
The system shall support at minimum 10,000 prediction requests per hour during business hours.

**NFR-5: Scalability**
The system shall scale linearly with the number of entities. Adding a new entity shall not require retraining the full backbone.

**NFR-6: Robustness**
The system shall produce well-calibrated predictions on out-of-distribution scenarios. Calibration shall be measured on held-out crisis periods (2008, 2020, 2022, SVB) and reliability diagrams shall be published per release.

**NFR-7: Privacy and Tenancy**
Customer-private data shall never enter the foundation training corpus. Customer fine-tuning shall occur in isolated tenant environments with no cross-tenant data flow.

### 3.3 Data Requirements

**DR-1: Coverage**
- Minimum 50,000 public entities (global)
- Minimum 20 years of quarterly history per entity (where available)
- Minimum 10 modalities (see § 5.6)

**DR-2: Quality**
- Point-in-time financial data with restatement tracking
- Source-of-truth identifiers (CIK, ISIN, LEI) for entity resolution
- Survivorship-bias-free (dead entities retained with delisting markers)

**DR-3: Graph Coverage**
- Supplier/customer relationships from 10-K disclosure + third-party datasets
- Competitor relationships from filings + market data + product taxonomies
- Ownership graph from beneficial ownership and corporate structure data
- Target: ≥ 60% edge coverage of material relationships in v1

### 3.4 Compliance and Trust Requirements

**CR-1: Explainability**
Every prediction surfaced to an end user shall have a human-readable explanation citing specific data, rules, and analogs. No "black-box" predictions in the user-facing layer.

**CR-2: Uncertainty Disclosure**
Confidence intervals (or full distributions) shall be displayed alongside every point prediction. Single-number predictions without uncertainty shall be prohibited in user-facing surfaces.

**CR-3: Model Risk Documentation**
The system shall maintain documentation compliant with SR 11-7 (US) and equivalent regulations: model purpose, design, data, performance monitoring, limitations, and approved use cases.

**CR-4: Override Mechanism**
Users with appropriate authority shall be able to override model predictions with documented reasons. Overrides shall be tracked and used as training signal.

---

## 4. System Architecture

### 4.1 High-Level View

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                                  USERS                                         │
│  Risk teams │ Strategy │ Audit │ Insurance │ PE/VC │ Treasury                 │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │ predictions, distributions,
                                       │ explanations, analogs
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                       APPLICATION / API LAYER                                  │
│         REST + streaming endpoints, tenant isolation, audit logging            │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                    PLANNING & DECISION INTERFACE  (Layer 5)                    │
│      Action enumeration │ scenario rollouts │ ranking │ analog retrieval       │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │
       ┌───────────────────────────────┼───────────────────────────────┐
       │                               │                               │
       ▼                               ▼                               ▼
┌──────────────┐              ┌──────────────────┐            ┌──────────────────┐
│ NEURO-SYMB.  │              │  DIFFUSION /     │            │  ANALOG STORE    │
│  LAYER (L4)  │              │  FLOW HEAD (L3)  │            │  (vector DB)     │
│              │              │                  │            │                  │
│ • constraints│              │ Distribution     │            │ Past states,     │
│ • rules      │              │ over future      │            │ KNN retrieval    │
│ • traces     │              │ latent states    │            │                  │
└──────────────┘              └────────┬─────────┘            └──────────────────┘
                                       │
                                       ▼
                            ┌──────────────────────┐
                            │ JEPA PREDICTION (L2) │
                            │ Point estimate of    │
                            │ future latent state  │
                            └────────┬─────────────┘
                                     │
                                     ▼
                            ┌──────────────────────┐
                            │ SSM BACKBONE  (L1)   │
                            │ Long-context encoder │
                            │ over entity history  │
                            └────────┬─────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                          DATA & FEATURE LAYER                                  │
│  Multimodal tokenizers │ entity resolution │ graph construction │ PIT engine   │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                      │
│ SEC EDGAR │ FRED │ GDELT │ market data │ insider trades │ news │ private data │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Component Map

```
Foundation Training            Inference                Customer Tenancy
─────────────────────          ─────────────            ────────────────
                                                        
[Data pipeline]                [API gateway]            [Private data]
      │                              │                        │
      ▼                              ▼                        ▼
[Pretraining cluster]          [Inference cluster]      [Fine-tune cluster]
      │                              │                        │
      ▼                              ▼                        ▼
[Foundation BWM]    ─────►     [Production BWM]   ◄───  [Customer BWM]
      │                              │                        
      ▼                              ▼                        
[Model registry]              [Explanation engine]            
                                     │                        
                                     ▼                        
                              [User-facing apps]              
```

---

## 5. Detailed Design

### 5.1 Layer 1: SSM Backbone

**Purpose:** Encode arbitrarily long per-entity histories with graph context, producing contextualized hidden states.

**Architecture choice:** hybrid SSM — Mamba-3 SSM blocks interleaved with full self-attention at a ~3:1 SSM:attention ratio, plus a separate sparse cross-attention adapter for graph-neighbor context. See § 5.1.1 (ADR-1) for the full decision record and rejected alternatives.

Justification (linear long-history scaling) still holds:
- 25 years × 4 quarters = 100 timesteps per entity, growing to 100 × N_neighbors with graph context
- SSM blocks remain the primary mixer; full self-attention is interleaved for precise event recall (a known SSM weakness) and cross-firm comparison

**Inputs:**
- Tokenized multimodal sequence per entity (numbers, text embeddings, event tokens, macro features)
- Graph neighbor states (suppliers, customers, competitors) via cross-attention adapters

**Outputs:**
- Contextualized hidden state `h_t` per (entity, timestep), dimension d_model = 1024 (target)

**Key design decisions:**

```
   Decision                       Choice                  Rationale
   ──────────────────────────────────────────────────────────────────────
   Backbone type                  Hybrid: Mamba-3 SSM     2026 SOTA consensus
                                  + full self-attention   (see § 5.1.1 / ADR-1);
                                  at ~3:1 SSM:attention   pure SSM weakness on
                                                          event recall + cross-
                                                          firm comparison
   Hidden dim                     1024                    Balance capacity vs cost
   Layers                         ~28 (sweep range)       Depth for hierarchy;
                                                          exact count from
                                                          Phase B sweep
   Total params                   ~500-700M (sweep)       Backbone ~300-350M;
                                                          Phase B sweeps
                                                          {150M, 350M, 700M}
   SSM block version              Mamba-3 (fallback       Exponential-trapezoidal
                                  Mamba-2 if kernels      discretization, complex-
                                  immature)               valued state (fixes
                                                          periodicity), MIMO
   Graph integration              Cross-attention         Sparse, learnable, kept
                                  adapters separate from  separate from temporal
                                  temporal self-attn      self-attention
   Position encoding              Quarter index + entity  Time-aware, identity-aware
                                  embedding
   Tokenization                   Mixed continuous-       Numbers stay continuous,
                                  discrete                text becomes discrete
                                                          tokens via T5 encoder
```

**Cross-entity graph handling:**

```
   Per-entity SSM blocks process entity history in parallel.
   Between blocks, a sparse cross-attention layer aggregates
   neighbor states weighted by edge type and edge weight:
   
   h'_t(A) = h_t(A) + Σ_{B ∈ N(A)} α_{AB,type} · W_type · h_t(B)
   
   Where:
   • N(A) is the graph neighborhood of A
   • α_{AB,type} is a learned edge weight by relationship type
   • W_type is a type-specific projection
```

### 5.1.1 ADR-1: SSM backbone — hybrid, not pure

**Decision.** Use a hybrid backbone: Mamba-3 SSM blocks as the primary sequence mixer, interleaved with full self-attention layers at roughly a 3:1 SSM:attention ratio, plus separate sparse cross-attention adapters for graph-neighbor context. Reject pure SSM.

**Status.** Accepted (spec update, May 2026 — ported from v3 ADR-1). Revisit if BWM moves to daily/intraday cadence.

**Context (2026 SOTA, researched May 2026).**

- Mamba-3 (Mar 2026) is the current pure-SSM frontier. Its three changes — exponential-trapezoidal discretization, complex-valued state, and MIMO — improve quality and halve state size, and the complex-valued state fixes the long-standing state-tracking weakness (parity, modular arithmetic, periodicity). BWM adopts the Mamba-3 block specifically for the periodicity benefit (seasonality, fiscal/refinancing cycles).
- The production consensus moved decisively to hybrid. 2026 flagship open-weight models (Qwen3.5, Kimi Linear, Ling 2.5, Nemotron 3, Falcon-H1) all interleave linear/SSM blocks with periodic full attention, commonly at a ~3:1 ratio. Qwen promoted its hybrid from a side-branch to the flagship line.
- Ablation evidence shows both component types are load-bearing: the linear/SSM block is the primary modeling backbone (catastrophic degradation when removed), while attention layers serve retrieval and stabilization. At least one team (MiniMax M2) reverted to full attention, citing poor reasoning/multi-turn accuracy from pure linear attention — a caution against an all-linear stack in production.
- The persistent SSM weakness is precise retrieval from context. This is the through-line across LLM, time-series, and the original Mamba-2 associative-recall results.

**Why this binds for BWM specifically.** Two BWM requirements land exactly on the SSM weakness: (a) precise recall of specific past events (a prior covenant breach, the last restatement, the firm's behavior in the last crisis), and (b) cross-firm comparison against a current peer set. Both are attention-favoring. BWM already requires cross-attention adapters for graph context, so attention is in the model regardless; adding temporal self-attention layers is a small marginal cost for a capability the business regime genuinely needs.

**Consequences.**

- Distributional head (Layer 3) is reinforced as mandatory: TS-SSM point forecasts show 8–18% mean error on standard benchmarks and require explicit uncertainty quantification to be decision-usable.
- The 3:1 ratio is a prior, not a constant — it was tuned for language at 30B–1T scale. BWM is ~600M on multimodal business time series, so the ratio and model size are Phase B sweep parameters (see § 6.3 acceptance criteria).
- Engineering-risk tradeoff: Mamba-3 kernels are recent (Mar 2026) and less battle-tested than Mamba-2. If Phase B tooling proves immature, fall back to a Mamba-2 + attention hybrid (the architecture decision holds; only the block version changes).

**Rejected alternatives.** Pure Mamba-3 (loses event recall + cross-firm reasoning; contradicts 2026 consensus). Pure transformer (loses the linear long-history scaling that is genuinely valuable at 100Q × ~200 neighbors; no efficiency argument for it here). Larger model "for capacity" (counterproductive in BWM's small-data regime — the binding constraint is data, not parameters).

### 5.2 Layer 2: JEPA Latent Prediction Head

**Purpose:** Learn the abstract latent state space and predict future latent states without reconstructing observations.

**Architecture:**

```
                ┌──────────────────┐         ┌──────────────────┐
   h_t  ──────► │ Online encoder   │         │ Target encoder   │ ◄─── h_{t+h}
                │ f_θ(h_t) = s_t   │         │ EMA of f_θ       │
                └────────┬─────────┘         └────────┬─────────┘
                         │                            │
                         ▼                            ▼
                       s_t                       s_{t+h}  (stop gradient)
                         │                            ▲
                         ▼                            │
                ┌──────────────────┐                  │
                │  Predictor g_φ   │── ŝ_{t+h} ───────┘
                │  (action a,      │
                │   horizon h)     │
                └──────────────────┘
                
   Loss:  L_JEPA = || ŝ_{t+h} − sg(s_{t+h}) ||²
                  + λ_var · VICReg variance term
                  + λ_cov · VICReg covariance term
```

**Key design decisions:**

```
   Decision                       Choice                  Rationale
   ──────────────────────────────────────────────────────────────────────
   Latent dimension               512                     Compress from 1024
   Target encoder update          EMA, τ = 0.996          Standard BYOL/JEPA
   Anti-collapse                  VICReg (var + cov)      Empirically robust
   Predictor                      Small transformer       Conditioning on (a, h)
                                  (6 layers)              needs flexibility
   Action encoding                Structured (categorical Actions are not free-
                                  + continuous)           form; finite types
   Horizon encoding               Learned embedding       Treat horizons as
                                  per discrete horizon    discrete tasks
```

**Action vocabulary (v1):**
- Acquisition, divestiture, IPO, capital raise (debt/equity)
- Layoff, hiring acceleration, executive change
- Price change, geographic expansion, product launch
- Material contract win/loss, regulatory event, restatement
- Macro shock (rate change, FX shock, commodity shock)

Actions are encoded as (type, magnitude, target_entity) tuples. The action vocabulary is extensible.

### 5.3 Layer 3: Diffusion / Flow Head

**Purpose:** Produce calibrated probability distributions over future latent states, not just point estimates.

**Architecture choice:** Conditional flow matching (continuous-time generative model). Justified by:
- Faster inference than diffusion (~4 steps vs ~50)
- Better calibration than classifier-free guidance on small latent spaces
- Stable training given the small dimensionality (512-d latent)

**Conditioning:**
- Current latent state `s_t`
- Action `a`
- Horizon `h`
- Optional macro/market context vector

**Output:** A sampler producing `s_{t+h} ~ p(s_{t+h} | s_t, a, h, context)`

**Training:**
- Frozen JEPA encoder produces latent states
- Flow head trained on (s_t, a, h, s_{t+h}) tuples from historical data
- Standard flow matching objective with classifier-free conditioning dropout

**Inference:**

```
   Single prediction:           Distribution (N=1000):
   ─────────────────────       ─────────────────────────
   Sample z ~ N(0,I)            Sample {z_1, ..., z_1000} ~ N(0,I)
   Solve ODE 4 steps            Solve ODE for each (batched)
   Return s_{t+h}               Return {s_{t+h}^(i)} distribution
                                + summary statistics (mean, quantiles)
```

**Key design decisions:**

```
   Decision                       Choice                  Rationale
   ──────────────────────────────────────────────────────────────────────
   Generative model              Conditional flow         Fast, stable on
                                 matching                 small latents
   ODE solver                    Euler with shortcuts     4 steps sufficient
   Conditioning method           Classifier-free          Standard, well-studied
                                 guidance                 
   Sample size                   1000 default,            P10/P50/P90 stable
                                 user-configurable        at N=500+
   Temperature                   Tunable per use case     Risk wants more spread
```

### 5.4 Layer 4: Neuro-Symbolic Layer

**Purpose:** Enforce hard constraints, encode soft rules as priors, and generate explanations.

**Two functions:**

1. **Constraint enforcement** — reject sampled futures from Layer 3 that violate hard rules
2. **Explanation generation** — produce symbolic traces of which rules and graph edges contributed to a prediction

**Architecture:**

```
   ┌─────────────────────────────────────────────────────────────────┐
   │                                                                 │
   │  HARD CONSTRAINTS (rejection sampling)                          │
   │  ─────────────────────────────────────                          │
   │  • assets = liabilities + equity                                │
   │  • cash_flow consistency identity                               │
   │  • all stock variables ≥ 0                                      │
   │  • predicted layoffs ≤ current headcount                        │
   │  • debt service ≤ available cash + new financing                │
   │                                                                 │
   │  Implementation: differentiable validators run on decoded       │
   │  latents; violations reject sample, valid samples continue.     │
   │                                                                 │
   ├─────────────────────────────────────────────────────────────────┤
   │                                                                 │
   │  SOFT RULES (priors and explanations)                           │
   │  ────────────────────────────────────                           │
   │  • supplier_concentration > 30% ∧ supplier_distress             │
   │    → downstream_margin_risk_up                                  │
   │  • debt/EBITDA > 6 ∧ rates_rising                               │
   │    → refinancing_risk_up                                        │
   │  • inventory_days ↑↑ ∧ revenue ↓                                │
   │    → margin_compression_imminent                                │
   │  • [hundreds more, organized by domain]                         │
   │                                                                 │
   │  Implementation: differentiable rule layer (Logic Tensor        │
   │  Network style). Each rule has a learned weight λ_rule that     │
   │  determines its influence on predictions and how strongly       │
   │  it appears in explanations.                                    │
   │                                                                 │
   ├─────────────────────────────────────────────────────────────────┤
   │                                                                 │
   │  EXPLANATION GENERATOR                                          │
   │  ─────────────────────                                          │
   │  For prediction (s_t, a) → s_{t+h}:                             │
   │  1. Decode s_t and s_{t+h} into interpretable variables         │
   │  2. Identify top-K rules that fired (highest λ × activation)    │
   │  3. Trace graph edges contributing via cross-attention weights  │
   │  4. Retrieve top-K analog historical cases                      │
   │  5. Compose into human-readable narrative + supporting data     │
   │                                                                 │
   └─────────────────────────────────────────────────────────────────┘
```

**Rule sourcing (combined approach):**

| Source | Type | Volume | Curation |
|---|---|---|---|
| Accounting standards | Hard | ~50-100 rules | Expert-coded, immutable |
| Regulatory definitions | Hard | ~100-300 rules | Expert-coded, jurisdiction-specific |
| Domain expert rules | Soft | ~500-2000 rules | Expert-authored, version-controlled |
| LLM-mined patterns | Soft (proposed) | ~1000-5000 candidates | LLM proposes, expert reviews |
| Data-mined associations | Soft (proposed) | ~5000+ candidates | Frequent pattern mining + expert review |

**Latent-to-interpretable decoder:**
A small auxiliary network maps the 512-d latent state onto a vocabulary of interpretable business variables (revenue regime, margin regime, leverage class, growth class, etc.). This is necessary because rules need to operate on interpretable quantities, not raw latents.

### 5.5 Layer 5: Planning and Decision Interface

**Purpose:** Translate a user's decision question into ranked, explained, distribution-aware recommendations.

**Core algorithm:**

```
   def plan(entity, current_state, candidate_actions, objective, horizon):
       results = []
       for action a in candidate_actions:
           # Step 1: Point estimate (fast)
           s_hat = jepa_head.predict(current_state, a, horizon)
           
           # Step 2: Distribution (slower, optional based on user tier)
           samples = diffusion_head.sample(current_state, a, horizon, n=1000)
           
           # Step 3: Validate against hard constraints
           valid = symbolic_layer.filter(samples)
           
           # Step 4: Decode to interpretable variables
           decoded = latent_decoder.decode(valid)
           
           # Step 5: Score per objective with uncertainty
           score = objective.evaluate(decoded)
           
           # Step 6: Generate explanation
           explanation = symbolic_layer.explain(current_state, s_hat, a)
           
           # Step 7: Retrieve analogs
           analogs = analog_store.knn(s_hat, k=5)
           
           results.append({
               action: a,
               distribution: decoded,
               score: score,
               explanation: explanation,
               analogs: analogs,
           })
       
       return rank(results, objective)
```

**Action candidate sources:**
- User-specified (explicit "what if we do X")
- Templated (common actions per entity type)
- Generated (model proposes actions in latent space)

**Objective specification:**
Objectives are user-defined functions over the decoded distribution. Examples:
- Maximize expected margin at H+8Q
- Minimize P10 of leverage ratio at H+4Q
- Maximize probability of staying above credit rating BBB

### 5.6 Data Pipeline

**Modalities:**

| Modality | Sources | Cadence | Tokenization |
|---|---|---|---|
| Financials | SEC EDGAR (XBRL), S&P, equiv. global | Quarterly | Continuous (normalized) |
| Filings text | EDGAR full-text, equiv. | Event-driven | Sentence-T5 embeddings |
| Earnings calls | Transcripts (Seeking Alpha, vendors) | Quarterly | Sentence-T5 embeddings |
| Market data | Yahoo, Polygon, exchange direct | Daily → quarterly aggr. | Continuous (returns, vol) |
| Macro | FRED, OECD, IMF | Monthly | Continuous |
| News | GDELT, vendor feeds | Daily | Event categorization + embedding |
| Insider trades | EDGAR Form 4, equiv. | Event-driven | Event tokens |
| Patents | USPTO, EPO | Quarterly aggregated | Counts + topic embeddings |
| Hiring data | Job postings, LinkedIn | Monthly aggregated | Continuous (counts, skill mix) |
| Supply chain | 10-K disclosure + vendors (Panjiva, etc.) | Event-driven | Graph edges |

**Point-in-time engine:**

```
   For training and backtesting, every data point is tagged with:
   • effective_date: when the event/observation occurred
   • availability_date: when this data became knowable to outsiders
   • revision_history: any subsequent restatements
   
   Queries during training: WHERE availability_date ≤ as_of_date
                            AND NOT was_subsequently_restated_before(as_of_date)
```

This is the single most operationally important data engineering task in the project. Get this wrong and the model's historical performance is fictional.

### 5.7 Graph Construction

**Relationship types:**

```
   Supplier ─► Customer    (directional, weight = revenue dependency)
   Competitor ◄─► Competitor  (bidirectional, weight = market overlap)
   Owner ─► Subsidiary     (directional, weight = ownership %)
   Lender ─► Borrower      (directional, weight = exposure)
   Insurer ─► Insured      (directional, weight = coverage)
```

**Sources:**
- Top-10 customer disclosures in 10-K
- Supply chain databases (Panjiva, Bloomberg SPLC)
- Patent citations, lawsuits
- Product taxonomies for competitor inference
- Corporate structure data for ownership

**Update cadence:** Edge weights are time-varying; graph is reconstructed quarterly.

**Missing edge handling:** The model is trained with random edge masking during pretraining so it is robust to incomplete graphs at inference.

---

## 6. Training Strategy

Training proceeds in ordered phases. Each phase produces a usable artifact; later phases assume earlier phases are frozen unless explicitly noted.

### 6.1 Phase Order

```
   Phase A: Data foundation
              │
              ▼
   Phase B: SSM + JEPA pretraining
              │
              ▼
   Phase C: Latent decoder + interpretability head
              │
              ▼
   Phase D: Diffusion / flow head
              │
              ▼
   Phase E: Symbolic layer integration
              │
              ▼
   Phase F: End-to-end fine-tuning (optional, risky)
              │
              ▼
   Phase G: Customer fine-tuning (per-tenant)
```

### 6.2 Phase A: Data Foundation

**Outputs:** Versioned, point-in-time, multimodal data lake; entity-resolved; graph-enriched.

**Acceptance criteria:**
- ≥ 50,000 entities with ≥ 80 quarters of financial data
- ≥ 10 modalities fused at consistent quarterly cadence
- Point-in-time integrity verified by spot-checks on historical events
- Graph coverage ≥ 60% on material relationships for top-1000 entities

### 6.3 Phase B: SSM + JEPA Pretraining

**Objective:** Self-supervised learning of regime-aware latent state.

**Loss:**

```
   L = L_JEPA  +  λ_var · L_VICReg_var  +  λ_cov · L_VICReg_cov
   
   Where:
   L_JEPA      = || ŝ_{t+h} − sg(s_{t+h}) ||²    (over multiple horizons)
   L_VICReg_var = max(0, γ − std(s_batch))²       (per dimension)
   L_VICReg_cov = sum of squared off-diagonal cov  (per dimension)
```

**Training regime:**
- Multi-horizon sampling: each batch has mixed (h=1Q, 2Q, 4Q, 8Q, 12Q)
- Random masking of input modalities (forces robustness)
- Random masking of graph edges (forces robustness)
- EMA target update with τ = 0.996

**Acceptance criteria:**
- Linear probe on regime classification: top-1 ≥ 70%
- Linear probe on margin direction (next 4Q): AUROC ≥ 0.75
- Linear probe on layoff event (next 4Q): AUROC ≥ 0.70
- Embedding-space neighbors qualitatively reviewed by domain experts on 50 sample entities
- **Architecture sweep complete** (per ADR-1 § 5.1.1): SSM:attention ratio ∈ {pure, 7:1, 3:1, 1:1} and model size ∈ {150M, 350M, 700M} evaluated on train/val gap; selected config documented in ADR-1.

### 6.4 Phase C: Latent Decoder

**Objective:** Train a small auxiliary network that maps 512-d latent to a vocabulary of interpretable business variables. This is required for the symbolic layer to operate.

**Decoder targets** (multitask):
- Revenue regime (8-class: high-growth, growth, mature, decline, recovery, stable, distressed, defaulted)
- Margin regime (5-class: expanding, stable, compressing, low, negative)
- Leverage class (5-class: under-levered, normal, elevated, high, distressed)
- Working capital regime (5-class)
- Each as a softmax head; backbone frozen

**Acceptance criteria:**
- Decoder accuracy ≥ 80% on held-out quarters across all classes

### 6.5 Phase D: Diffusion / Flow Head

**Objective:** Learn `p(s_{t+h} | s_t, a, h, context)`.

**Training:**
- Frozen SSM + JEPA encoder produces (s_t, s_{t+h}) pairs
- Conditional flow matching with classifier-free guidance dropout (10%)
- Train on all historical (state, action, future state) tuples plus null-action (passive forecasting)

**Acceptance criteria:**
- Calibration: 90% predicted intervals contain 88-92% of actual outcomes
- Sharpness: P10-P90 spread ≤ 1.5× empirical spread on stable regimes
- OOD calibration on 2008/2020 held-out: 90% CI contains ≥ 75% of outcomes (degradation expected, not collapse)

### 6.6 Phase E: Symbolic Layer Integration

**Two parallel tracks:**

**E1: Hard constraints**
- Implement differentiable validators for accounting identities and sign constraints
- Integrate as rejection-sampling filter on Layer 3 outputs
- Acceptance: 100% of returned samples pass all hard constraints

**E2: Soft rules**
- Author initial ~500 rule rules with domain experts
- Generate ~5000 candidate rules via LLM + data mining
- Curate down to ~2000 reviewed rules
- Train differentiable rule layer to align with JEPA predictions where they agree
- Generate explanation traces

**Acceptance criteria:**
- 95% of predictions have an explanation with ≥ 3 substantive rule citations
- Expert review: 80% of explanations rated "useful and accurate" on a 100-sample audit

### 6.7 Phase F: End-to-End Fine-Tuning (Optional)

**Objective:** Joint optimization across all layers with composite loss.

**Loss:**

```
   L_total = L_JEPA + λ₁ · L_diffusion + λ₂ · L_constraint + λ₃ · L_decoder
```

**Cautions:**
- High risk of destabilizing pretrained representations
- Symbolic constraints are partially non-differentiable; surrogate gradients required
- Small learning rate (1e-6 range); short fine-tune duration
- Strong validation gating: roll back if any acceptance criterion from prior phases regresses

**This phase is optional.** v1 ships without it. It is included for completeness and as a research direction for future versions.

### 6.8 Phase G: Customer Fine-Tuning

**Objective:** Adapt the foundation BWM to a customer's private data and use case.

**Approach:**
- Frozen foundation model
- LoRA adapters on the SSM backbone for backbone fine-tune
- Custom rules and constraints layer for customer-specific business logic
- Customer-specific entities added to graph
- Per-tenant data isolation; no cross-tenant gradient flow

---

## 7. Evaluation Framework

### 7.1 Benchmark Suite ("BWM-Bench")

A held-out benchmark across multiple historical periods and tasks. This is intended to be open-sourced as a contribution to the field.

**Held-out periods (entirely excluded from training):**
- 2008-2009 financial crisis
- 2020 COVID shock
- 2022 rate hike cycle
- SVB collapse (March 2023)
- Energy crisis (2022-2023)

**Tasks:**

| Task | Metric | Baseline | Target (v1) |
|---|---|---|---|
| Next-Q revenue regime | Top-1 accuracy | XGBoost on raw features | +15% absolute |
| Margin direction (4Q) | AUROC | Industry-naive logistic | +0.10 |
| Layoff event (4Q) | AUROC | Sector-average baseline | ≥ 0.75 |
| Default event (12Q) | AUROC | Altman Z-score | ≥ 0.80 |
| Drawdown >20% (12Q) | AUROC | Volatility baseline | ≥ 0.70 |
| Crisis-period calibration | Reliability diagram | N/A | Within ±5% of perfect |
| Counterfactual fidelity | Backtested A/B events | Pre/post analysis | Directionally correct on 70% of test cases |

### 7.2 Calibration Monitoring

Per release, publish:
- Reliability diagrams across all probability tasks
- P10/P50/P90 coverage on continuous predictions
- OOD performance degradation curves
- Per-sector, per-region, per-size breakdowns

### 7.3 Explanation Quality

Recurring expert audits on stratified samples:
- 100 explanations per quarter
- Rated on accuracy, completeness, relevance
- Target: 80% rated "useful and accurate"

### 7.4 Decision Impact (Production)

For deployed customers:
- Track prediction → action → outcome chains
- Compare model-recommended actions to actual actions
- Measure realized value where attribution is possible
- Use as training signal in Phase G fine-tunes

---

## 8. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Point-in-time data errors invalidate all backtests | Critical | Aggressive PIT engineering; spot-check historical events; third-party audit |
| Phase F (end-to-end) destabilizes pretrained reps | High | Treat Phase F as optional; ship v1 without it |
| Symbolic rule authoring is bottleneck | High | LLM-assisted drafting; tiered review; ship with smaller rule set initially |
| Graph data is too incomplete to be useful | High | Train with edge masking; degrade gracefully when neighbors missing |
| Foundation model captures noise instead of signal | High | VICReg + multi-horizon objective + heavy linear-probe validation gates |
| Customer fine-tuning leaks signal across tenants | Critical | Strict tenant isolation; per-tenant adapter weights; no shared gradients |
| OOD performance collapses in regime change | Medium | Train on multiple historical crises; explicit OOD calibration tests |
| Compute cost exceeds projections | Medium | Phased training keeps Phase B as only large run; later phases cheaper |
| Regulator rejects opacity of latent layer | Medium | Symbolic layer + decoder + explanations specifically designed for this |
| Adversarial inputs (manipulated filings) | Medium | Anomaly detection on inputs; flag rather than predict on suspicious data |

---

## 9. Open Design Questions

These are decisions deferred to design refinement rather than fixed in this spec.

**OQ-1: Graph representation granularity.** Are subsidiary entities separate nodes or rolled up? Does the graph include people (executives, board members) as nodes?

**OQ-2: Text encoder choice.** Sentence-T5 is the default for filings/calls, but a domain-finetuned encoder (FinBERT-style) may outperform. Trade-off: marginal accuracy vs maintenance burden.

**OQ-3: Action vocabulary closure.** The initial action set is finite. How do we handle novel actions (e.g., a new type of financial instrument)? Open vocabulary via LLM tokenization, or strict typed schema?

**OQ-4: Multi-step rollout.** Beyond 12Q horizon, do we autoregress (use predicted s_{t+h} as input for s_{t+2h}) or train direct long-horizon predictors? Autoregression compounds error; direct prediction is data-sparse at long horizons.

**OQ-5: Cross-jurisdiction entity resolution.** Same company, different identifiers (CIK vs LEI vs ISIN). How aggressively do we merge?

**OQ-6: Real-time vs batched inference.** Most use cases are batched, but live event response (news shock → immediate update) is occasionally requested. Architecture must support both modes.

**OQ-7: Counterfactual identification strategy.** Pure observational counterfactuals are unreliable. Do we explicitly model treatment assignment (propensity scores, instrumental variables) or rely on the latent space + analog retrieval to surface uncertainty in counterfactual claims?

---

## 10. Glossary

- **BWM**: Business World Model — this system
- **PIT**: Point-in-time, referring to data integrity where only information knowable on date D is used for predictions about date D
- **SSM**: State-space model, a class of sequence models with linear scaling (Mamba, S4)
- **JEPA**: Joint Embedding Predictive Architecture, a self-supervised paradigm that predicts in latent space rather than reconstructing observations
- **VICReg**: Variance-Invariance-Covariance Regularization, a method to prevent representation collapse in self-supervised learning
- **EMA**: Exponential Moving Average, used here for the target encoder in JEPA-style training
- **Flow matching**: A continuous-time generative modeling technique, alternative to diffusion
- **LoRA**: Low-Rank Adaptation, a parameter-efficient fine-tuning method
- **Regime**: A qualitative state of a business entity (e.g., "high-growth," "distressed") that is more stable than the underlying numerical metrics

---

## Appendix A: Architecture Diagram (Detailed)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              USER / API LAYER                                    │
└──────────────────────────────────────┬───────────────────────────────────────────┘
                                       │ requests: (entity, action?, horizon, obj)
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       LAYER 5: PLANNING INTERFACE                                │
│  Enumerate actions │ orchestrate L1-L4 │ rank │ retrieve analogs │ format        │
└──────────────────────────────────────┬───────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────────┐
        │                              │                                  │
        ▼                              ▼                                  ▼
┌─────────────────────┐    ┌─────────────────────┐         ┌─────────────────────┐
│ ANALOG STORE        │    │ LAYER 4: SYMBOLIC   │         │ LAYER 3: DIFFUSION  │
│                     │    │                     │         │                     │
│ Vector DB of past   │    │ ┌─ Hard constraints │         │ Conditional flow    │
│ entity states       │    │ ├─ Soft rules       │         │ matching head       │
│ KNN retrieval       │    │ ├─ Rule trace gen   │         │                     │
│                     │    │ └─ Latent decoder   │         │ Samples N futures   │
└─────────────────────┘    └──────────┬──────────┘         └──────────┬──────────┘
                                      │                               │
                                      └───────────┬───────────────────┘
                                                  │
                                                  ▼
                              ┌─────────────────────────────────┐
                              │ LAYER 2: JEPA PREDICTION HEAD   │
                              │                                 │
                              │ Online encoder + EMA target +   │
                              │ action-conditioned predictor    │
                              │                                 │
                              │ Outputs: s_t (current latent),  │
                              │          ŝ_{t+h} (point pred)   │
                              └──────────────┬──────────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────────┐
                              │ LAYER 1: HYBRID BACKBONE        │
                              │                                 │
                              │ Mamba-3 SSM + interleaved full  │
                              │ self-attn (~3:1) + sparse graph │
                              │ cross-attention adapters        │
                              │ (see ADR-1 / § 5.1.1)           │
                              │                                 │
                              │ Inputs: tokenized multimodal    │
                              │ history per entity + neighbors  │
                              │ Outputs: h_t hidden states      │
                              └──────────────┬──────────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────────┐
                              │ DATA & FEATURE LAYER            │
                              │                                 │
                              │ • Point-in-time engine          │
                              │ • Entity resolution             │
                              │ • Multimodal tokenizers         │
                              │ • Graph construction            │
                              │ • Feature versioning            │
                              └──────────────┬──────────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────────┐
                              │ DATA SOURCES                    │
                              │                                 │
                              │ Financials │ Text │ Market │    │
                              │ Macro │ News │ Events │ Graph  │
                              └─────────────────────────────────┘
```

---

## Appendix B: Inference Flow (Detailed)

```
USER QUERY: "If we acquire CompanyX at $5B, what's our 8Q outlook?"

Step 1: Parse and validate
─────────────────────────
  entity = Y (asker)
  action = {type: ACQUISITION, target: X, price: 5e9}
  horizon = 8 quarters
  objective = user-supplied scoring function

Step 2: Fetch context (data layer)
─────────────────────────────────
  Pull Y's history (last 80Q), X's history (last 80Q),
  graph neighbors of both (top-50 each), macro context
  
Step 3: Encode (Layer 1)
────────────────────────
  SSM backbone processes all histories with graph cross-attention
  Output: h_t(Y), h_t(X), h_t(neighbors)

Step 4: Compute current latent (Layer 2 encoder)
────────────────────────────────────────────────
  s_t(Y) = f_θ(h_t(Y))

Step 5: Point estimate (Layer 2 predictor)
──────────────────────────────────────────
  ŝ_{t+8}(Y) = g_φ(s_t(Y), action, h=8)

Step 6: Distribution (Layer 3)
─────────────────────────────
  samples = diffusion_head.sample(s_t(Y), action, h=8, n=1000)
  → {s_{t+8}^(1), ..., s_{t+8}^(1000)}

Step 7: Filter (Layer 4 hard constraints)
─────────────────────────────────────────
  valid_samples = symbolic.filter(samples)
  → typically 800-950 retained

Step 8: Decode (Layer 4 latent decoder)
───────────────────────────────────────
  decoded = decoder(valid_samples)
  → distributions over: revenue, margin, leverage, headcount,
                       market_share, regime classification

Step 9: Score (Layer 5)
──────────────────────
  score = objective(decoded)
  → scalar + distribution over score

Step 10: Explain (Layer 4)
──────────────────────────
  explanation = symbolic.explain(s_t(Y), ŝ_{t+8}(Y), action)
  → ranked rule chains with cited data and graph edges

Step 11: Retrieve analogs (analog store)
────────────────────────────────────────
  analogs = vector_db.knn(ŝ_{t+8}(Y), k=5)
  → 5 historical cases of similar predicted states + actual outcomes

Step 12: Format response
────────────────────────
  Return: {
    point_estimate: decoded(ŝ_{t+8}(Y)),
    distribution: {P10, P50, P90} per variable,
    explanation: ranked_rules,
    analogs: [{company, year, similarity, outcome}, ...],
    confidence: calibration_score,
    warnings: [OOD_flags, missing_data_flags, ...]
  }
```

---

*End of specification.*
