# Business World Model (BWM) — Technical Specification, v3

**Position.** BWM is the first conformant application of the Norn world-modeling conventions on the Uni embedded neuro-symbolic database. It is an L1 Predictor and L2 Simulator (in the sense of Chu et al. 2026) for the business-economic regime — a hybrid governing-law regime spanning digital, physical, social, and scientific constraints.

**Scope discipline.** BWM does not extend or fork Uni; it uses Uni's public Cypher/Locy/Python/REST surface. BWM does not author Norn conventions; it consumes them and surfaces gaps as Norn proposals. Where Uni does not yet provide a capability BWM needs, the spec documents a feature request rather than building below the public surface.

**Status.** v3 design specification. Supersedes v2 in framing and decomposition; preserves the underlying L1/L2/L3 × governing-law analysis.

**Revision note (2026-05, v3.1).** Backbone updated from Mamba-2 to a hybrid Mamba-3 + attention design following a review of 2026 SSM SOTA; see ADR-1 (§ 4.6). Affects § 4.1, § 4.2, § 4.6, and § 9.2. No change to capability targets or layer decomposition.

**Audience.** BWM engineering and research, Norn and Uni maintainers (for feature-request items), product, executive sponsors.

---

## 0. How v3 differs from v2

v2 specified BWM as a vertically integrated stack including data layer, latent prediction layer, distributional layer, governing-law layer, and planning layer. v3 retains the same capability targets (L1 in v1, L1+distributional in v2, L2 in v3, partial L3 in v4+) but reorganizes responsibilities now that the substrate is concrete.

Specifically, several pieces v2 specified as custom BWM components are now Uni primitives consumed via the public API:

- v2's "differentiable rule layer (LTN-style)" → Locy rules with PROB columns and monotonic probability aggregators (`MNOR`, `MPROD`)
- v2's "hard-constraint validator (rejection sampling)" → Locy rules with stratified negation
- v2's "Layer 5 planning loop that orchestrates rollouts" → `ASSUME { ... } THEN { ... }` issued as a Locy command
- v2's "inverse-dynamics anomaly detection" → `ABDUCE` over observed state deltas, complementing a neural inverse-dynamics head
- v2's "custom point-in-time engine with effective_date / availability_date / revision_history" → BTIC properties + Uni snapshots + `VERSION AS OF` / `TIMESTAMP AS OF`
- v2's "entity reconciliation across CIK/ticker/LEI/vendor IDs" → UniId content addressing + per-label `ext_id`
- v2's "MREP version locking" → Uni named snapshots
- v2's "multi-source merge for vendor data conflicts" → Uni CRDT property types

The architectural thesis from v2 stands: a good simulator looks more like the constraints than like the world. With Locy as the constraint engine, this thesis stops being aspirational and becomes the operating discipline.

---

## 1. Position

### 1.1 The business-economic regime

Chu et al. (2026) organize the world-modeling field along two axes: capability level (L1 Predictor / L2 Simulator / L3 Evolver) and governing-law regime (physical / digital / social / scientific). None of the four regimes is "business-economic." BWM occupies that empty cell, and the cell is structurally hybrid:

```
                         Formalizability
                        (low ←──→ high)
                  ┌──────────────────────────────────┐
                  │                                  │
        High      │   Social ★               Digital │
   Observability  │   ── strategic         ★ ── accounting
                  │      interaction,         identities,
                  │      ToM about            disclosures,
                  │      competitors          XBRL schemas
                  │                                  │
                  │                                  │
                  │                                  │
        Low       │   Scientific ★          Physical │
   Observability  │   ── firm causal      ★ ── supply
                  │      mechanisms,          chain flows,
                  │      market dynamics      capacity,
                  │      (empirical)          inventory
                  │                            conservation
                  └──────────────────────────────────┘
```

Each business decision touches multiple sub-regimes simultaneously. A merger affects accounting (digital), production capacity (physical), competitive dynamics (social), and demand mechanisms (scientific). The architecture must support enforcement of formal identities, conservation of physical quantities, modeling of strategic interaction, and empirical validation of partially understood mechanisms — within a single composed query.

This is the design constraint that drove the v2 architecture and continues to drive v3.

### 1.2 Capability targets

```
   Version    Capability                          Status
   ────────────────────────────────────────────────────────
   v1         L1 Predictor (state inference,      Future
              forward dynamics, observation
              decoding) for single-firm
              short-horizon prediction
   
   v2         L1 + inverse dynamics +             Future
              distributional head; calibrated
              uncertainty across horizons
   
   v3         L2 Simulator passing all three      Future
              boundary conditions (long-horizon
              coherence, intervention sensitivity,
              constraint consistency) for the
              business regime
   
   v4+        Partial L3 Evolver: evidence-       Future,
              driven model revision triggered     conditional
              by customer deployment outcomes;    on Norn L3
              regression-gated updates            spec maturing
```

L1/L2/L3 are not a static classification — they describe the capability invoked at any moment. A deployed BWM operates at L1 for fast lookups, L2 for scenario simulation, and (eventually) L3 when systematic prediction failures accumulate.

### 1.3 What BWM is not

```
   N-1   Not a price prediction system. BWM predicts
         business state. Trading is out of scope.
   
   N-2   Not L3 in v1-v3. True L3 (active information
         expansion, persistent revision, governed
         validation) is v4+. Per-tenant fine-tuning is
         protective-belt updating, not hard-core
         revision.
   
   N-3   Not a closed-loop deployment. Closed-loop use
         (planning in interaction with a live
         environment) is an orthogonal deployment
         property of consumer applications, not of the
         model itself.
   
   N-4   Not a source of causal claims. BWM surfaces
         hypotheses, analogs, and uncertainty grounded
         in observational data. It supports human or
         experimental verification; it does not assert
         causation.
   
   N-5   Not a single-entity model. Predictions require
         neighbor states (suppliers, customers,
         competitors, macro). Isolated single-firm use
         is unsupported.
   
   N-6   Not a frontier general-purpose LLM. BWM is a
         domain model. Text is encoded via pretrained
         language models; BWM does not produce general
         text output.
```

---

## 2. Boundaries

This section enumerates explicitly what BWM owns, what it depends on from Norn, and what it consumes from Uni. The discipline is: when in doubt, the capability lives lower in the stack, not in BWM.

### 2.1 What BWM owns

```
   ┌────────────────────────────────────────────────────────────┐
   │ BWM owns                                                   │
   │ ──────────                                                 │
   │                                                            │
   │ A. Domain ontology                                         │
   │    • Business entity labels (Firm, Filing, Period,         │
   │      Event, Action, Macro, Industry, Person, Supply        │
   │      relationship, Lender relationship, ...)               │
   │    • Action vocabulary for the business regime             │
   │    • event_kind registry conforming to Norn CPTE-style     │
   │      conventions where applicable                          │
   │                                                            │
   │ B. Learned models, registered with uni-xervo               │
   │    • Firm-state encoder (SSM backbone)                     │
   │    • JEPA forward-dynamics predictor                       │
   │    • Inverse-dynamics head                                 │
   │    • Diffusion distributional head                         │
   │    • Supporting fine-tuned classifiers (sentiment,         │
   │      restatement detector, sector reclassifier, etc.)      │
   │    • Foundation pretraining pipeline for all of the        │
   │      above                                                 │
   │                                                            │
   │ C. Locy rule library                                       │
   │    • bwm.accounting — hard identities                      │
   │    • bwm.regulation — hard regulatory constraints          │
   │    • bwm.empirics — soft mechanism rules with PROB         │
   │    • bwm.strategy — competitive interaction rules          │
   │    • bwm.supplychain — propagation rules                   │
   │                                                            │
   │ D. Data pipeline                                           │
   │    • EDGAR + global filing equivalents                     │
   │    • FRED + macro data                                     │
   │    • Market data ingestion                                 │
   │    • Earnings call transcripts                             │
   │    • News and event streams                                │
   │    • Supply-chain disclosure + vendor feeds                │
   │    • Insider trade filings                                 │
   │    • Patent and hiring signals                             │
   │    • All point-in-time engineering                         │
   │                                                            │
   │ E. BWM-Bench                                               │
   │    • Task catalog                                          │
   │    • Held-out crisis-period evaluations                    │
   │    • Failure taxonomy detectors                            │
   │    • MREP conformance package                              │
   │                                                            │
   │ F. Customer integration patterns                           │
   │    • Tenancy via Locy module composition                   │
   │    • Per-tenant fine-tuning (LoRA on registered models)    │
   │    • Audit and explanation surfacing                       │
   │                                                            │
   └────────────────────────────────────────────────────────────┘
```

### 2.2 What BWM uses from Norn

```
   ┌────────────────────────────────────────────────────────────┐
   │ BWM consumes from Norn                                     │
   │ ───────────────────────                                    │
   │                                                            │
   │ A. The L1/L2/L3 × governing-law vocabulary and its         │
   │    operationalization (ASR, COD, the three L2              │
   │    boundary conditions)                                    │
   │                                                            │
   │ B. CPTE-style metadata conventions (belief, perspective,   │
   │    policy_tags, causal_links, source_certainty,            │
   │    valid_from / valid_to) for events that should be        │
   │    auditable as world-state evidence                       │
   │                                                            │
   │ C. The MREP (Minimal Reproducible Evaluation Package)      │
   │    standard — BWM-Bench is structured as an MREP           │
   │    instance for the business regime                        │
   │                                                            │
   │ D. The "look like the constraints, not the world"          │
   │    architectural thesis                                    │
   │                                                            │
   │ E. The Lakatosian hard-core / protective-belt              │
   │    distinction for model revision (relevant in v4+)        │
   │                                                            │
   │ BWM does NOT extend Norn from inside the BWM spec. Where   │
   │ Norn does not yet specify something BWM needs, BWM either  │
   │ implements it internally (with a note) or files a Norn     │
   │ proposal.                                                  │
   └────────────────────────────────────────────────────────────┘
```

### 2.3 What BWM uses from Uni

```
   ┌────────────────────────────────────────────────────────────┐
   │ BWM consumes from Uni                                      │
   │ ──────────────────────                                     │
   │                                                            │
   │ Storage and query                                          │
   │   • Labeled property graph with Cypher                     │
   │   • Lance-backed columnar storage on S3/GCS/local          │
   │   • LSM-style write path with auto-compaction              │
   │   • Snapshot isolation + WAL                               │
   │                                                            │
   │ Identity and types                                         │
   │   • VID for execution; UniId for content-addressed         │
   │     cross-system identity; ext_id for primary keys         │
   │   • BTIC (Binary Temporal Interval Codec) for              │
   │     point-in-time fields with epistemic certainty         │
   │   • CRDT types (8 variants) for multi-source merge         │
   │   • Full Arrow type system                                 │
   │                                                            │
   │ Indexing                                                   │
   │   • B-tree, hash, bitmap scalar indexes                    │
   │   • Vector indexes (HNSW, IVF-PQ, Flat)                    │
   │   • BM25 full-text and JSON-path indexes                   │
   │   • Inverted indexes                                       │
   │                                                            │
   │ Neuro-symbolic primitives                                  │
   │   • similar_to() / ~= for vector similarity in queries     │
   │   • Auto-embedding via Candle (uni-xervo)                  │
   │   • Cross-encoder reranking via uni-xervo                  │
   │   • Model invocation inside Cypher and Locy via            │
   │     uni-xervo aliases (classifiers, generation,            │
   │     custom ONNX)                                           │
   │   • Hybrid search via reciprocal rank fusion               │
   │                                                            │
   │ Reasoning (Locy)                                           │
   │   • Recursive rules with stratified negation               │
   │   • Semi-naive fixpoint evaluation                         │
   │   • PROB columns and probabilistic IS NOT                  │
   │   • MNOR / MPROD aggregators with shared-proof             │
   │     detection and optional BDD-based exact mode            │
   │   • ALONG / FOLD / BEST BY / PRIORITY                      │
   │   • DERIVE for rule-driven graph mutation                  │
   │   • ASSUME { ... } THEN { ... } for hypothetical           │
   │     reasoning with rollback                                │
   │   • ABDUCE for "what would have to change" queries         │
   │   • EXPLAIN RULE for derivation traces                     │
   │   • Module system with USE                                 │
   │                                                            │
   │ Time travel                                                │
   │   • VERSION AS OF '<snapshot_id>'                          │
   │   • TIMESTAMP AS OF '<datetime>'                           │
   │   • Named snapshots for milestone version locking          │
   │                                                            │
   │ Graph algorithms (36+ in uni-algo)                         │
   │   • PageRank, Louvain, shortest path, betweenness,         │
   │     etc. — used for supply-chain centrality, market        │
   │     structure analysis, etc.                               │
   │                                                            │
   │ APIs and tooling                                           │
   │   • Rust crate (uni-db)                                    │
   │   • Python bindings (uni-db PyO3)                          │
   │   • Pydantic OGM (uni-pydantic) for type-safe              │
   │     entity modeling                                        │
   │   • CLI for bulk import, snapshot admin, etc.              │
   │                                                            │
   │ BWM consumes all of the above via public APIs only.        │
   │ BWM does not depend on Uni internals.                      │
   │                                                            │
   └────────────────────────────────────────────────────────────┘
```

### 2.4 Requirements BWM has of Uni

These are documented as expectations BWM will rely on. Where current Uni does not yet satisfy them, the item is flagged as a feature request. The intent is that the BWM design assumes Uni will meet these; if a requirement is unmet, BWM either waits or implements above the public surface.

```
   Req     Description                                Status
   ────────────────────────────────────────────────────────────
   UR-1    Locy stratified evaluator scales to       Verify
           ~50K entities × 25 years × ~5 rule         (UR-1)
           modules with combined rule count
           ~2500 within Phase F training-loop
           latency budget (target: full
           rule-set evaluation ≤ 60s on
           representative BWM corpus)
   
   UR-2    BTIC values are projectable to             Verify
           numeric tensor features at the             (UR-2)
           neural encoder boundary via a
           documented pattern (start_offset,
           duration, certainty_one_hot, plus
           granularity). May require BWM-side
           helper utilities; Uni provides the
           accessor functions.
   
   UR-3    exact_probability with shared-proof        Verify
           detection has a documented cost           (UR-3)
           model so BWM can decide per-query
           whether to opt in. BWM default is
           exact_probability=false; specific
           high-stakes queries (e.g., bank
           stress testing) opt in with a
           bounded max_bdd_variables.
   
   UR-4    uni-xervo can host BWM-trained             Confirmed
           SSM, JEPA, and diffusion models            via docs
           via the standard alias catalog.            (uni-xervo
           ONNX export pipeline assumed for          supports raw
           BWM model deployment.                      ONNX)
   
   UR-5    Snapshot creation and restoration          Confirmed
           is fast enough to support MREP             via Part XII
           version locking on the BWM corpus
           (target: snapshot creation ≤ 30s
           on corpus of ~50K entities × 25Y).
   
   UR-6    Locy modules support per-tenant            Verify
           composition: a customer-specific           (UR-6)
           module can USE bwm.empirics and
           override specific rules without
           forking the base module.
   
   UR-7    Cross-encoder reranking via                Confirmed
           uni-xervo applies to analog                via Part VIII
           retrieval queries (similar_to)
           against firm-state embeddings.
   
   UR-8    Time-travel queries (VERSION AS OF /       Confirmed
           TIMESTAMP AS OF) work correctly             via Part XII
           through Locy rule evaluation, so
           BWM-Bench can re-run rule
           evaluations against historical
           snapshots.
```

Items marked "Verify" go in § 10 as open questions.

---

## 3. The business-regime data model

This section specifies the labels, edge types, and key properties that constitute BWM's domain ontology. The model is expressed in Uni's labeled property graph; CPTE-style metadata fields (per Norn conventions) appear on event-type labels.

### 3.1 Core entity labels

```
   ┌────────────────────────────────────────────────────────────┐
   │                                                            │
   │   Firm                                                     │
   │     ext_id: CIK or equivalent (LEI fallback)               │
   │     name: String                                           │
   │     incorporation_jurisdiction: String                     │
   │     incorporation_date: BTIC                               │
   │     primary_industry_code: String (NAICS or GICS)          │
   │     status_window: BTIC (active, dissolved, acquired)      │
   │     firm_state_embedding: Vector[1024]  (from BWM SSM)     │
   │     [HNSW index on firm_state_embedding for analog         │
   │      retrieval]                                            │
   │                                                            │
   │   Listing                                                  │
   │     ext_id: <exchange>:<ticker>                            │
   │     primary_exchange: String                               │
   │     trading_window: BTIC                                   │
   │     [edge IS_LISTING_OF → Firm]                            │
   │                                                            │
   │   Filing                                                   │
   │     ext_id: SEC accession number or equivalent             │
   │     event_kind: 'FILING'  (Norn CPTE convention)           │
   │     form_type: String  ('10-K', '10-Q', '8-K', etc.)       │
   │     period_covered: BTIC                                   │
   │     filed_at: Timestamp                                    │
   │     knowable_from: Timestamp  (PIT availability)           │
   │     belief: Float  (1.0 unless restated)                   │
   │     perspective: String  (regulatory body)                 │
   │     policy_tags: GSet[String]                              │
   │     causal_links: List[UniId]  (e.g., supersedes a         │
   │       previous filing for restatements)                    │
   │     financials: Map(String → Float)  (line items)          │
   │     prepared_remarks_text: String                          │
   │     prepared_remarks_embedding: Vector[768]                │
   │     [edge HAS_FILING from Firm]                            │
   │                                                            │
   │   Period                                                   │
   │     ext_id: <firm_ext_id>:<period_iso>                     │
   │     period: BTIC  (e.g., btic('2024-Q3'))                  │
   │     period_type: String  ('quarter' | 'year' | 'month')    │
   │     financials_consolidated: Map(String → Float)           │
   │     restatement_history: List[UniId]                       │
   │     [edge OF_FIRM → Firm]                                  │
   │                                                            │
   │   Event                                                    │
   │     ext_id: source-specific event id                       │
   │     event_kind: String  (Norn CPTE convention; see § 3.3)  │
   │     payload: Map  (event-specific structured fields)       │
   │     valid_from: BTIC                                       │
   │     recorded_at: Timestamp                                 │
   │     knowable_from: Timestamp                               │
   │     belief: Float | BeliefDist  (CRDT-backed where merge   │
   │       across sources is needed)                            │
   │     perspective: String                                    │
   │     policy_tags: GSet[String]                              │
   │     causal_links: List[UniId]                              │
   │     source_certainty: Float                                │
   │     event_embedding: Vector[768]  (for analog retrieval)   │
   │     [edges AFFECTS → Firm; OCCURRED_AT → Period]           │
   │                                                            │
   │   Action                                                   │
   │     ext_id: <firm_ext_id>:<action_kind>:<announcement_ts>  │
   │     action_kind: String  (see action vocabulary § 3.4)     │
   │     announcement_date: Timestamp                           │
   │     effective_window: BTIC                                 │
   │     parameters: Map  (e.g., {target: <firm>, value: $5B}   │
   │       for an acquisition)                                  │
   │     status: String  ('announced' | 'completed' |           │
   │       'cancelled' | 'pending')                             │
   │     [edges TAKEN_BY → Firm; TARGETS → Firm (optional);     │
   │      RESULTED_IN_FILING → Filing (optional)]               │
   │                                                            │
   │   Macro                                                    │
   │     ext_id: <series_code>  (e.g., 'FEDFUNDS', 'GDP')       │
   │     series: String                                         │
   │     value_at: Map(BTIC → Float)  (or as separate           │
   │       Observation nodes — see § 3.2)                       │
   │     [edge OBSERVED_AT → Period]                            │
   │                                                            │
   │   Industry                                                 │
   │     ext_id: NAICS or GICS code                             │
   │     name: String                                           │
   │     parent_industry: String  (hierarchy)                   │
   │     [edges BELONGS_TO → Industry; CONTAINS → Firm]         │
   │                                                            │
   │   Person                                                   │
   │     ext_id: source-specific (LinkedIn, EDGAR insider id)   │
   │     name: String                                           │
   │     [edges ROLE_AT → Firm with role/start/end properties]  │
   │                                                            │
   └────────────────────────────────────────────────────────────┘
```

### 3.2 Observation pattern for time series

For high-frequency time series (macro, market data, daily news counts), per-period values are stored as separate `Observation` nodes rather than as map fields on the series node, so that BTIC + snapshot semantics apply to individual readings and restatements can be modeled as new observations superseding old ones.

```
   Observation
     ext_id: <series>:<period_iso>
     series_ext_id: String
     period: BTIC
     value: Float | Map  (depending on series)
     belief: Float
     source: String
     [edge OF_SERIES → Macro; edge SUPERSEDES → Observation
      (for revisions)]
```

This pattern is borrowed from CPTE-style event modeling and is exactly the kind of convention that should eventually live in Norn rather than BWM.

### 3.3 event_kind registry for the business regime

Initial set; expected to grow during Phase A data engineering. Each is a registered `event_kind` value on `Event` and `Filing` nodes, conforming to Norn CPTE-style metadata expectations.

```
   Category               event_kind values
   ──────────────────────────────────────────────────────────
   Filing                 FILING_10K, FILING_10Q, FILING_8K,
                          FILING_DEF14A, FILING_S1, FILING_FOREIGN
   
   Corporate action       ANNOUNCEMENT_ACQUISITION,
                          ANNOUNCEMENT_DIVESTITURE,
                          ANNOUNCEMENT_IPO,
                          ANNOUNCEMENT_CAPITAL_RAISE_DEBT,
                          ANNOUNCEMENT_CAPITAL_RAISE_EQUITY,
                          ANNOUNCEMENT_LAYOFF,
                          ANNOUNCEMENT_HIRING_PLAN,
                          ANNOUNCEMENT_EXEC_CHANGE,
                          ANNOUNCEMENT_PRICE_CHANGE,
                          ANNOUNCEMENT_MARKET_ENTRY,
                          ANNOUNCEMENT_PRODUCT_LAUNCH,
                          ANNOUNCEMENT_CONTRACT_MATERIAL,
                          ANNOUNCEMENT_RESTRUCTURING
   
   Regulatory             REGULATORY_INVESTIGATION,
                          REGULATORY_SANCTION,
                          REGULATORY_RULE_CHANGE,
                          REGULATORY_APPROVAL,
                          REGULATORY_LITIGATION
   
   Macro                  MACRO_RATE_DECISION,
                          MACRO_RELEASE,
                          MACRO_SHOCK
   
   Restatement            RESTATEMENT_FILING,
                          RESTATEMENT_GUIDANCE
   
   Earnings               EARNINGS_REPORT,
                          EARNINGS_GUIDANCE,
                          EARNINGS_CALL
   
   Insider                INSIDER_BUY, INSIDER_SELL,
                          INSIDER_OPTION_GRANT
   
   News                   NEWS_MATERIAL  (with sub-classification
                          via event_embedding + classifier)
```

Each `event_kind` has a registered shape schema (validated at write time) and an associated soft-rule context (used by `bwm.empirics` to attach probabilistic priors).

### 3.4 Action vocabulary

Closed initial set for v1-v3. Extension requires explicit registration. Defined as a controlled vocabulary on the `Action` label's `action_kind` property.

```
   action_kind                Parameters
   ──────────────────────────────────────────────────────────
   acquire                    target: Firm, value: Float,
                              cash_pct: Float, stock_pct: Float
   
   divest                     unit_ext_id: String,
                              proceeds: Float
   
   ipo                        proceeds_target: Float,
                              listing_exchange: String
   
   raise_debt                 amount: Float, coupon_pct: Float,
                              maturity_years: Int
   
   raise_equity               amount: Float, dilution_pct: Float
   
   layoff                     headcount_change: Int  (negative)
   
   hire                       headcount_change: Int  (positive)
   
   change_exec                role: String, departing: Person,
                              arriving: Person
   
   change_price               product_family: String,
                              pct_change: Float
   
   enter_market               geographic: String,
                              capex_required: Float
   
   launch_product             product_family: String,
                              expected_revenue: Float
   
   restructure                description: String,
                              one_time_charge: Float
   
   declare_dividend           per_share: Float,
                              total_payout: Float
   
   buyback                    amount_authorized: Float
   
   null_action                (no-op; baseline for COD)
```

The `null_action` is structurally important: it is the baseline against which Counterfactual Outcome Deviation (COD, see § 6.2) is computed for every candidate action.

### 3.5 Key edge types

```
   Edge type             From → To              Properties
   ──────────────────────────────────────────────────────────
   HAS_FILING            Firm → Filing          —
   OF_FIRM               Period → Firm          —
   IS_LISTING_OF         Listing → Firm         —
   AFFECTS               Event → Firm           strength: Float,
                                                 belief: Float
   OCCURRED_AT           Event → Period         —
   TAKEN_BY              Action → Firm          —
   TARGETS               Action → Firm          —
   RESULTED_IN_FILING    Action → Filing        —
   BELONGS_TO            Firm → Industry        weight: Float
   CONTAINS              Industry → Industry    —
   ROLE_AT               Person → Firm          role: String,
                                                 window: BTIC
   
   SUPPLIES              Firm → Firm            concentration:
                                                 Float (pct of
                                                 supplier's
                                                 revenue),
                                                 window: BTIC,
                                                 belief: Float
                                                 (often
                                                 incomplete data)
   
   CUSTOMER_OF           Firm → Firm            (inverse of
                                                 SUPPLIES)
   
   COMPETES_WITH         Firm → Firm            strength: Float
                                                 (overlapping
                                                 product
                                                 markets),
                                                 belief: Float
   
   LENDS_TO              Firm → Firm            amount: Float,
                                                 window: BTIC,
                                                 belief: Float
   
   HOLDS_EQUITY_IN       Firm → Firm            pct: Float,
                                                 window: BTIC
   
   OBSERVED_AT           Observation → Period   —
   OF_SERIES             Observation → Macro    —
   SUPERSEDES            <any> → <same>         reason: String
   
   ANALOG_OF             Firm → Firm            similarity:
                                                 Float,
                                                 computed_at:
                                                 Timestamp
                                                 (materialized
                                                 via Locy
                                                 DERIVE from
                                                 vector
                                                 similarity)
```

### 3.6 BTIC usage pattern

BTIC is BWM's canonical type for any field that represents an interval with potentially uncertain bounds. The conventions:

```
   Field semantics              BTIC encoding
   ──────────────────────────────────────────────────────────
   Quarterly reporting period   btic('2024-Q3')
                                  → [2024-07-01, 2024-10-01)
                                  granularity: quarter
                                  certainty: definite
   
   Restated value's original    btic(start='2024-07-01',
   validity window                    end='2024-10-15',
                                       lo_certainty=definite,
                                       hi_certainty=approximate)
                                  → ends when restatement
                                    published, with fuzzy
                                    boundary
   
   Effective date of an         btic(point='2024-11-15',
   announced layoff (not yet         certainty=approximate)
   begun)                         → instantaneous, uncertain
                                    because announcements
                                    are often re-timed
   
   Period of a private equity   btic(start='2023-06-01',
   firm's ownership of a              end=null,
   target                              lo_certainty=definite,
                                       hi_certainty=unknown)
                                  → ownership ongoing,
                                    end unknown
   
   Historical regime period     btic(start='2008-Q3',
   for held-out evaluation           end='2009-Q2',
                                       granularity=quarter,
                                       certainty=definite)
                                  → financial crisis window
                                    for MREP held-out
                                    evaluation
```

The combination of `valid_from` (BTIC), `knowable_from` (Timestamp), and `belief` on every event/filing/observation enforces point-in-time integrity at the data model level. A query at `TIMESTAMP AS OF '2024-08-15'` automatically filters to data knowable on that date.

---

## 4. BWM's learned models

This section specifies the models BWM trains, how they're packaged for uni-xervo, and how they're invoked from Cypher and Locy. The integration surface is new versus v2 (models are uni-xervo aliases callable from rules), and the backbone is updated to a hybrid SSM design per ADR-1 (§ 4.6), reflecting 2026 SOTA in which flagship long-context models converged on SSM/linear + periodic full attention rather than pure SSM.

### 4.1 Model catalog

```
   ┌────────────────────────────────────────────────────────────┐
   │                                                            │
   │   Alias                              Purpose                │
   │   ──────────────────────────────────────────────────────    │
   │                                                            │
   │   bwm/firm-state-encoder/v1         Map firm history to    │
   │                                     1024-d state vector;   │
   │                                     hybrid Mamba-3 + attn  │
   │                                     backbone (~3:1 SSM:    │
   │                                     attention) with sparse │
   │                                     graph cross-attn.      │
   │                                                            │
   │   bwm/forward-dynamics-jepa/v1      Predict z_{t+h} given  │
   │                                     z_t and action a;      │
   │                                     online encoder + EMA   │
   │                                     target + predictor.    │
   │                                                            │
   │   bwm/inverse-dynamics/v1           Predict a from         │
   │                                     (z_t, z_{t+1});        │
   │                                     diagnostic for         │
   │                                     restatement / fraud    │
   │                                     detection (composed    │
   │                                     with ABDUCE).          │
   │                                                            │
   │   bwm/diffusion-head/v1             Conditional flow       │
   │                                     matching head producing │
   │                                     distributions over     │
   │                                     z_{t+h}.               │
   │                                                            │
   │   bwm/latent-decoder/v1             p(o | z) decoders for  │
   │                                     interpretable regimes  │
   │                                     (revenue, margin,      │
   │                                     leverage, growth,      │
   │                                     working-capital).      │
   │                                                            │
   │   bwm/guidance-tone/v2              Sentiment / tone       │
   │                                     classifier on prepared │
   │                                     remarks text; fine-    │
   │                                     tuned LM head.         │
   │                                                            │
   │   bwm/restatement-detector/v1       Binary classifier      │
   │                                     flagging filings       │
   │                                     likely to be restated. │
   │                                                            │
   │   bwm/event-embedding/v1            Text-to-vector for     │
   │                                     news and filing        │
   │                                     content; sentence-T5   │
   │                                     fine-tuned.            │
   │                                                            │
   │   bwm/sector-reclassifier/v1        Predict GICS sector    │
   │                                     drift from financial   │
   │                                     state changes.         │
   │                                                            │
   └────────────────────────────────────────────────────────────┘
```

Each alias resolves to an ONNX export deployed via uni-xervo's catalog (assumption per UR-4). Versioning follows semver; older versions remain available so that snapshots referencing them remain reproducible.

### 4.2 Training stack (high-level)

BWM owns the training pipeline end-to-end; uni-xervo is the deployment target, not the training environment.

```
   Component              Stack
   ──────────────────────────────────────────────────────────
   Backbone               Hybrid SSM (see ADR-1, § 4.6): Mamba-3
                          blocks (exponential-trapezoidal
                          discretization, complex-valued state,
                          MIMO) interleaved with full self-
                          attention at a ~3:1 SSM:attention ratio
                          for event recall, plus sparse graph
                          cross-attention adapters kept separate
                          from temporal attention. 1024 hidden,
                          ~28 layers, ~300-350M backbone within a
                          ~500-700M total. Attention ratio and
                          model size are swept in Phase B, not
                          fixed. PyTorch + custom CUDA for
                          selective scan.
   
   JEPA training          Standard JEPA loss (||ŝ_{t+h} -
                          sg(s_{t+h})||² + VICReg) on multi-
                          horizon (h ∈ {1,2,4,8,12} quarters)
                          targets. Add inverse-dynamics
                          auxiliary loss.
   
   Diffusion head         Conditional flow matching, ~50M
                          params, 4-step inference target.
                          Trained on residuals between JEPA
                          point estimates and observed
                          latents.
   
   Decoders               Light MLP heads, one per regime
                          variable, trained jointly with
                          backbone or separately depending on
                          phase.
   
   Classifiers            Sentence-T5-base or DeBERTa-v3-base
                          fine-tunes for text classification
                          tasks; <100M params each.
```

Foundation pretraining is the heaviest compute commitment. Detailed budget and schedule are deferred to a separate training plan; this spec specifies only the interfaces.

### 4.3 Invocation patterns

The key insight from the Black Book is that BWM models are callable inside Cypher and Locy via uni-xervo. This collapses what v2 specified as application-layer composition into in-query operations.

**Pattern A: model invocation inside a Cypher MATCH**

```cypher
// Find filings whose tone classifier disagrees with their
// reported guidance — candidate restatement signals.

MATCH (f:Firm)-[:HAS_FILING]->(filing:Filing)
WHERE filing.form_type = '10-Q'
  AND filing.knowable_from <= $as_of
  AND classify('bwm/guidance-tone/v2', filing.prepared_remarks_text)
        IN ['cautious', 'defensive']
  AND filing.guidance_direction = 'positive'
RETURN f.ext_id, filing.ext_id, filing.period_covered
```

**Pattern B: similarity search using a BWM-trained embedding**

```cypher
// Find historical firms most similar to a target firm's
// current state, restricted to firms that subsequently
// experienced distress.

MATCH (target:Firm {ext_id: $target_ext_id})
MATCH (analog:Firm)
WHERE analog.ext_id <> target.ext_id
  AND similar_to(analog.firm_state_embedding,
                 target.firm_state_embedding) > 0.85
  AND analog IS distress_outcome_within_8q
RETURN analog.ext_id,
       analog.subsequent_outcome,
       analog.firm_state_embedding <=> target.firm_state_embedding
         AS distance
ORDER BY distance ASC
LIMIT 10
```

**Pattern C: model invocation inside a Locy rule**

```cypher
MODULE bwm.empirics

CREATE RULE adverse_tone_with_margin_compression AS
    MATCH (f:Firm)-[:HAS_FILING]->(filing:Filing)
    WHERE filing.gross_margin_qoq_change < -200
      AND classify('bwm/guidance-tone/v2',
                   filing.prepared_remarks_text)
            IN ['cautious', 'defensive']
    YIELD KEY f, 0.68 AS guidance_cut_prob PROB
```

**Pattern D: forward-dynamics prediction inside ASSUME**

```cypher
// Counterfactual rollout: what does Firm X look like in 4 quarters
// IF it announces an acquisition of size $5B?

ASSUME {
    MATCH (f:Firm {ext_id: $target_ext_id})
    CREATE (f)-[:TAKES_ACTION]->(:Action {
      action_kind: 'acquire',
      parameters: {target: $acquirer_target, value: 5000}
    })
} THEN {
    MATCH (f:Firm {ext_id: $target_ext_id})
    WITH f, predict(
      'bwm/forward-dynamics-jepa/v1',
      f.firm_state_embedding,
      {horizon: 4, action: 'acquire', value: 5000}
    ) AS predicted_state_4q
    RETURN f.ext_id, predicted_state_4q
}
```

The savepoint+rollback in ASSUME ensures the synthetic action does not pollute the actual graph. This is structurally cleaner than v2's application-layer enumeration loop.

### 4.4 Model versioning and snapshot coupling

Because models are invoked by alias, and aliases can resolve to specific versions, BWM-Bench evaluations couple to model versions explicitly:

```
   BWM-Bench run protocol
   ──────────────────────
   1. Pin Uni snapshot: VERSION AS OF '<bench-snapshot-id>'
   2. Pin model versions:
        bwm/firm-state-encoder/v1.2.3
        bwm/forward-dynamics-jepa/v1.4.0
        bwm/diffusion-head/v0.9.1
        ...
   3. Pin Locy module versions:
        bwm.accounting@v1.3
        bwm.empirics@v2.1
        ...
   4. Execute task suite
   5. Record full provenance triple
      (snapshot_id, model_versions, module_versions)
```

This is part of MREP conformance (see § 8.2).

### 4.5 Inverse dynamics: dual neural + symbolic

BWM v2 added inverse dynamics as the fourth L1 operator. v3 implements it twice:

```
   Mode                                    Use
   ──────────────────────────────────────────────────────────
   Neural inverse dynamics                 Continuous-state
   (bwm/inverse-dynamics/v1):              setting; runs at
   given z_{t-1} and z_t, predict          inference time as
   a distribution over actions in          a fast probe.
   the action vocabulary.
   
   Symbolic abductive inverse              Discrete-action
   dynamics (Locy ABDUCE on                setting; gives
   observed state delta):                  minimal-change
   given Firm.state_at_t-1 and             explanation for
   Firm.state_at_t, ABDUCE the              audit; runs at
   minimum-change set of Actions           query time on
   that would explain the delta.           specific cases.
```

The two are complementary, not redundant. The neural version is faster and continuous; the symbolic version is slower but produces auditable explanations. Both feed the same downstream "this state change is inconsistent with logged actions" alert, but the symbolic version is what gets surfaced in audit trails.

### 4.6 ADR-1: SSM backbone — hybrid, not pure

**Decision.** Use a hybrid backbone: Mamba-3 SSM blocks as the primary sequence mixer, interleaved with full self-attention layers at roughly a 3:1 SSM:attention ratio, plus separate sparse cross-attention adapters for graph-neighbor context. Reject pure SSM.

**Status.** Accepted (v3). Revisit if BWM moves to daily/intraday cadence.

**Context (2026 SOTA, researched May 2026).**

- Mamba-3 (Mar 2026) is the current pure-SSM frontier. Its three changes — exponential-trapezoidal discretization, complex-valued state, and MIMO — improve quality and halve state size, and the complex-valued state fixes the long-standing state-tracking weakness (parity, modular arithmetic, periodicity). BWM adopts the Mamba-3 block specifically for the periodicity benefit (seasonality, fiscal/refinancing cycles).
- The production consensus moved decisively to hybrid. 2026 flagship open-weight models (Qwen3.5, Kimi Linear, Ling 2.5, Nemotron 3, Falcon-H1) all interleave linear/SSM blocks with periodic full attention, commonly at a ~3:1 ratio. Qwen promoted its hybrid from a side-branch to the flagship line.
- Ablation evidence shows both component types are load-bearing: the linear/SSM block is the primary modeling backbone (catastrophic degradation when removed), while attention layers serve retrieval and stabilization. At least one team (MiniMax M2) reverted to full attention, citing poor reasoning/multi-turn accuracy from pure linear attention — a caution against an all-linear stack in production.
- The persistent SSM weakness is precise retrieval from context. This is the through-line across LLM, time-series, and the original Mamba-2 associative-recall results.

**Why this binds for BWM specifically.** Two BWM requirements land exactly on the SSM weakness: (a) precise recall of specific past events (a prior covenant breach, the last restatement, the firm's behavior in the last crisis), and (b) cross-firm comparison against a current peer set. Both are attention-favoring. BWM already requires cross-attention adapters for graph context, so attention is in the model regardless; adding temporal self-attention layers is a small marginal cost for a capability the business regime genuinely needs.

**Consequences.**

- Distributional head (Layer 3) is reinforced as mandatory: TS-SSM point forecasts show 8-18% mean error on standard benchmarks and require explicit uncertainty quantification to be decision-usable.
- The 3:1 ratio is a prior, not a constant — it was tuned for language at 30B-1T scale. BWM is ~600M on multimodal business time series, so the ratio and model size are Phase B sweep parameters (see § 9.2).
- Engineering-risk tradeoff: Mamba-3 kernels are recent (Mar 2026) and less battle-tested than Mamba-2. If Phase B tooling proves immature, fall back to a Mamba-2 + attention hybrid (the architecture decision holds; only the block version changes).

**Rejected alternatives.** Pure Mamba-3 (loses event recall + cross-firm reasoning; contradicts 2026 consensus). Pure transformer (loses the linear long-history scaling that is genuinely valuable at 100Q × ~200 neighbors; no efficiency argument for it here). Larger model "for capacity" (counterproductive in BWM's small-data regime — the binding constraint is data, not parameters).

---

## 5. The Locy rule library

BWM authors five rule modules. Each lives in version control alongside model artifacts; each is versioned independently; each declares dependencies on Uni features explicitly.

### 5.1 Module catalog

```
   ┌────────────────────────────────────────────────────────────┐
   │ Module                Purpose                       Size   │
   │ ─────────────────────────────────────────────────────────  │
   │                                                            │
   │ bwm.accounting        Hard accounting identities    ~80    │
   │                       and sign constraints          rules  │
   │                                                            │
   │ bwm.regulation        Hard regulatory constraints   ~40    │
   │                       (jurisdiction-specific)       rules  │
   │                                                            │
   │ bwm.empirics          Soft empirical mechanism      ~500-  │
   │                       rules with PROB columns       2000   │
   │                                                     rules  │
   │                                                            │
   │ bwm.strategy          Competitive-interaction       ~300   │
   │                       rules (ToM-style for          rules  │
   │                       business)                            │
   │                                                            │
   │ bwm.supplychain       Propagation rules through     ~300   │
   │                       supply / customer / lender    rules  │
   │                       graph                                │
   │                                                            │
   │ bwm.tenant.<id>       Customer-specific overrides   varies │
   │                       via Locy USE composition             │
   │                                                            │
   └────────────────────────────────────────────────────────────┘
```

### 5.2 bwm.accounting — hard identities

These rules return Boolean. Violations filter samples during constrained generation in § 6.

```cypher
MODULE bwm.accounting

// Balance sheet identity
CREATE RULE balance_sheet_holds AS
    MATCH (f:Firm)-[:HAS_FILING]->(filing:Filing)
    WHERE abs(filing.financials['assets']
              - filing.financials['liabilities']
              - filing.financials['equity']) < 0.01
            * filing.financials['assets']
    YIELD KEY filing

// Cash-flow identity
CREATE RULE cash_flow_identity_holds AS
    MATCH (filing:Filing)
    WHERE filing.financials['cash_change']
          = filing.financials['op_cash_flow']
          + filing.financials['inv_cash_flow']
          + filing.financials['fin_cash_flow']
    YIELD KEY filing

// Sign and domain constraints
CREATE RULE sign_constraints_hold AS
    MATCH (filing:Filing)
    WHERE filing.financials['revenue'] >= 0
      AND filing.financials['headcount'] >= 0
      AND filing.financials['shares_outstanding'] > 0
    YIELD KEY filing

// Working capital identity
CREATE RULE working_capital_identity AS
    MATCH (filing:Filing)
    WHERE abs(filing.financials['working_capital']
              - (filing.financials['current_assets']
                 - filing.financials['current_liabilities'])) < 0.01
    YIELD KEY filing
```

These rules are used in two contexts: (a) as data validation during ingest (a filing failing `balance_sheet_holds` is flagged for human review, not rejected — restatements occasionally fail identity checks until corrected), and (b) as hard constraints during generation in § 6.

### 5.3 bwm.regulation — jurisdiction-aware constraints

These rules are parameterized by jurisdiction (`policy_tags` filter) and version (PRIORITY-ordered).

```cypher
MODULE bwm.regulation

// Bank: Common Equity Tier 1 minimum (Basel III)
CREATE RULE basel_iii_cet1_minimum [PRIORITY 100] AS
    MATCH (f:Firm)-[:HAS_FILING]->(filing:Filing)
    WHERE f.primary_industry_code STARTS WITH '522'  // banking
      AND filing.policy_tags @> ['JURISDICTION_US']
      AND filing.financials['cet1_ratio'] >= 0.045
    YIELD KEY f

// Insurance: NAIC RBC minimum
CREATE RULE naic_rbc_minimum [PRIORITY 100] AS
    MATCH (f:Firm)-[:HAS_FILING]->(filing:Filing)
    WHERE f.primary_industry_code STARTS WITH '524'  // insurance
      AND filing.policy_tags @> ['JURISDICTION_US']
      AND filing.financials['rbc_ratio'] >= 2.0
    YIELD KEY f

// ... ~40 such rules across major jurisdictions and industries
```

PRIORITY ordering matters here: a federal rule supersedes a state rule for the same constraint; later versions of the same rule supersede earlier versions. Locy's PRIORITY semantics handle this directly.

### 5.4 bwm.empirics — soft mechanism rules

This is the largest module and the locus of most curation work. Rules carry PROB columns that combine via MNOR when multiple signals fire on the same firm.

```cypher
MODULE bwm.empirics

// Margin compression precedes guidance cuts
CREATE RULE margin_compression_signal AS
    MATCH (f:Firm)-[:HAS_PERIOD]->(p:Period)
    WHERE p.financials_consolidated['gross_margin_qoq_change'] < -200
    YIELD KEY f, KEY p, 0.45 AS guidance_cut_prob PROB

// Inventory growth without revenue growth
CREATE RULE inventory_buildup_signal AS
    MATCH (f:Firm)-[:HAS_PERIOD]->(p:Period)
    WHERE p.financials_consolidated['inventory_days_qoq_change'] > 30
      AND p.financials_consolidated['revenue_yoy_change'] < 0.02
    YIELD KEY f, KEY p, 0.38 AS margin_compression_next_q_prob PROB

// Composite: combine margin-compression and inventory signals
// MNOR is monotonic and safe in recursive strata.
CREATE RULE pre_guidance_cut_warning AS
    MATCH (f:Firm)-[:HAS_PERIOD]->(p:Period)
    WHERE (f, p) IS margin_compression_signal
       OR (f, p) IS inventory_buildup_signal
    FOLD warning_score = MNOR(p.signal_prob)
    YIELD KEY f, KEY p, warning_score PROB

// Rate-sensitive refinancing risk
CREATE RULE refinancing_risk_when_rates_rising AS
    MATCH (f:Firm)-[:HAS_FILING]->(filing:Filing)
    MATCH (m:Macro {ext_id: 'FEDFUNDS'})-[:OBSERVED_AT]->(p:Period)
    WHERE filing.financials['debt_to_ebitda'] > 6.0
      AND p.financials_consolidated['yoy_change'] > 1.0  // rate hike
      AND filing.financials['debt_maturity_within_1y'] > 0.25
            * filing.financials['total_debt']
    YIELD KEY f, 0.62 AS refi_risk PROB

// Rule that uses a BWM model inside its WHERE clause
CREATE RULE adverse_tone_signal AS
    MATCH (f:Firm)-[:HAS_FILING]->(filing:Filing)
    WHERE classify('bwm/guidance-tone/v2',
                   filing.prepared_remarks_text)
            IN ['cautious', 'defensive']
    YIELD KEY f, KEY filing,
          confidence('bwm/guidance-tone/v2',
                     filing.prepared_remarks_text)
            AS adverse_tone_prob PROB
```

The combination of model invocation inside rules and PROB-column composition is the actual neuro-symbolic primitive of this stack. There is no separate "fusion" layer.

### 5.5 bwm.strategy — competitive interaction

Strategic interaction is the social regime for business. Rules here encode Theory-of-Mind-style reasoning over competitors.

```cypher
MODULE bwm.strategy

// Competitor entered our market → margin pressure expected
CREATE RULE competitor_market_entry_pressure AS
    MATCH (incumbent:Firm)-[c:COMPETES_WITH]-(entrant:Firm)
    MATCH (a:Action {action_kind: 'enter_market'})-[:TAKEN_BY]->(entrant)
    WHERE c.strength > 0.5
      AND a.announcement_date >= $window_start
    YIELD KEY incumbent,
          c.strength * 0.4 AS margin_pressure_4q_prob PROB

// Regulatory action on one firm signals industry exposure
CREATE RULE regulatory_action_industry_propagation AS
    MATCH (target:Firm)<-[:AFFECTS]-(e:Event {event_kind:
                                              'REGULATORY_INVESTIGATION'})
    MATCH (target)-[:BELONGS_TO]->(industry:Industry)
    MATCH (peer:Firm)-[:BELONGS_TO]->(industry)
    WHERE peer.ext_id <> target.ext_id
    YIELD KEY peer,
          0.3 AS regulatory_compliance_cost_increase_prob PROB

// Executive moves between competitors are strategic signals
CREATE RULE executive_competitor_transfer AS
    MATCH (p:Person)-[r1:ROLE_AT {role: 'CEO'}]->(former:Firm)
    MATCH (p)-[r2:ROLE_AT]->(new:Firm)
    MATCH (former)-[c:COMPETES_WITH]-(new)
    WHERE r1.window.hi <= r2.window.lo  // BTIC ordering
      AND btic_duration(btic_gap(r1.window, r2.window)) < 365 * 86400000
    YIELD KEY new, 0.55 AS strategic_shift_2q_prob PROB
```

### 5.6 bwm.supplychain — propagation

Recursive rules through the supplier/customer/lender graph. These benefit most from Locy's stratified fixpoint with semi-naive evaluation.

```cypher
MODULE bwm.supplychain

// Direct supplier distress propagates to customer
CREATE RULE supplier_distress_propagates AS
    MATCH (customer:Firm)-[s:SUPPLIES]->(supplier:Firm)
    WHERE supplier IS distressed
      AND s.concentration > 0.15
    YIELD KEY customer,
          s.concentration * 0.6 AS supply_disruption_prob PROB

// Recursive: multi-hop supply-chain risk
// MNOR combines independent paths; shared-proof detection
// flags when two paths share intermediate suppliers.
CREATE RULE upstream_disruption_chain AS
    MATCH (customer:Firm)-[s:SUPPLIES]->(intermediate:Firm)
    WHERE intermediate IS upstream_disruption_chain
       OR intermediate IS supplier_distress_propagates
    ALONG path_strength = prev.path_strength * s.concentration
    FOLD risk = MNOR(intermediate.path_risk)
    BEST BY path_strength DESC
    YIELD KEY customer, risk AS supply_disruption_prob PROB

// Lender exposure to a defaulting borrower
CREATE RULE lender_loss_propagation AS
    MATCH (lender:Firm)-[l:LENDS_TO]->(borrower:Firm)
    WHERE borrower IS defaulted
      AND l.amount > 0.05 * lender.financials['total_assets']
    YIELD KEY lender, 
          (l.amount / lender.financials['total_assets']) * 0.7
            AS lender_loss_prob PROB
```

The `upstream_disruption_chain` rule is recursive and uses MNOR aggregation. Per the Black Book, MNOR is monotonic and safe in recursive strata. Shared-proof detection will flag groups where two supply paths share intermediate suppliers; for those, BWM defaults to the independence-assumption result and surfaces the `_approximate=true` annotation in the explanation trace.

### 5.7 Rule sourcing and curation

```
   Source                          Type    Curation
   ────────────────────────────────────────────────────────
   GAAP / IFRS / XBRL accounting   Hard    Expert, sourced
   taxonomies                              from authoritative
                                           documentation
   
   Bank / insurance / pharma        Hard    Expert, by
   regulatory minimums                     jurisdiction
   
   Industry-expert business         Soft    Expert review;
   rules (CFOs, sector analysts,           initial set ~500;
   risk managers)                          target ~2000
   
   LLM-mined candidate rules        Soft    LLM proposes from
   from analyst reports                    text corpus; expert
                                           reviews and tunes
                                           PROB weights
   
   Data-mined associations          Soft    Frequent-pattern
                                           mining on historical
                                           outcomes; expert
                                           review before
                                           inclusion
```

Initial rule weights for soft rules are bootstrapped from historical conditional probabilities; subsequent refinement comes from BWM-Bench performance and customer feedback. Rule versions are tracked in git; each module has a semver version that BWM-Bench runs pin.

### 5.8 Tenant composition

Per-tenant customization is achieved through Locy's module system without forking base modules.

```cypher
MODULE bwm.tenant.acme_bank
USE bwm.accounting { balance_sheet_holds, cash_flow_identity_holds }
USE bwm.regulation { basel_iii_cet1_minimum }
USE bwm.empirics { *  EXCEPT margin_compression_signal }
USE bwm.supplychain { lender_loss_propagation }

// Tenant-specific override: ACME bank uses tighter
// CET1 threshold than regulatory minimum
CREATE RULE acme_internal_cet1 [PRIORITY 200] AS
    MATCH (f:Firm)-[:HAS_FILING]->(filing:Filing)
    WHERE filing.financials['cet1_ratio'] >= 0.08  // ACME internal target
    YIELD KEY f
```

PRIORITY ordering ensures the tenant override applies before the base rule. This pattern depends on UR-6 (Locy module composition with per-rule override).

---

## 6. The decision interface

This is the surface that downstream agents (risk teams, strategy teams, audit, insurance underwriting, PE/VC screening) interact with. Every query is a Cypher or Locy expression; every response carries a provenance trace.

### 6.1 Primary metrics: ASR and COD

Decision-grade evaluation requires action-success-oriented metrics, not prediction-accuracy metrics. Per Chu et al. and v2 § 11.2:

```
   ASR (Action Success Rate)
     ASR = (1/N) Σ 𝟙[task_i succeeds under policy
                     derived from p̂]
     
     Tests: long-horizon coherence + decision-usability
   
   COD(k) (Counterfactual Outcome Deviation at step k)
     COD(k) = E[d(ẑ_H^(a1), ẑ_H^(a2))]
              where a1, a2 differ at step k
     
     Tests: intervention sensitivity
   
   CVR (Constraint Violation Rate)
     Hard CVR ≡ 0 by construction (rejection sampling
                  via bwm.accounting + bwm.regulation)
     Soft CVR  = fraction of returned samples violating
                  any bwm.empirics rule with PROB > 0.8
     
     Tests: constraint consistency
```

### 6.2 Layer 5 planning as ASSUME orchestration

The earlier-versions "Layer 5 planning loop" disappears as application-layer code. It becomes a sequence of `ASSUME { ... } THEN { ... }` queries, one per candidate action plus a null-action baseline. The diff between rollouts is COD.

```python
# Python orchestration layer — BWM's role is to enumerate
# candidate actions and assemble queries. The execution
# is Uni's.

def evaluate_actions_for_firm(
    session: uni_db.Session,
    firm_ext_id: str,
    candidate_actions: list[Action],
    horizon_quarters: int,
    objective: Objective,
    snapshot_id: str,
) -> list[ActionEvaluation]:
    results = []
    
    # Baseline: null action
    null_outcome = session.execute_locy(
        f"""
        VERSION AS OF '{snapshot_id}'
        ASSUME {{
            // null action: no change
        }} THEN {{
            MATCH (f:Firm {{ext_id: $firm_ext_id}})
            WITH f, predict(
              'bwm/diffusion-head/v1',
              f.firm_state_embedding,
              {{horizon: {horizon_quarters},
                action: 'null_action',
                samples: 500}}
            ) AS rollout
            RETURN rollout
        }}
        """,
        params={'firm_ext_id': firm_ext_id},
    )
    
    for action in candidate_actions:
        outcome = session.execute_locy(
            f"""
            VERSION AS OF '{snapshot_id}'
            ASSUME {{
                MATCH (f:Firm {{ext_id: $firm_ext_id}})
                CREATE (f)-[:TAKES_ACTION]->(:Action {{
                  action_kind: $action_kind,
                  parameters: $params
                }})
            }} THEN {{
                MATCH (f:Firm {{ext_id: $firm_ext_id}})
                WITH f, predict(
                  'bwm/diffusion-head/v1',
                  f.firm_state_embedding,
                  {{horizon: {horizon_quarters},
                    action: $action_kind,
                    parameters: $params,
                    samples: 500}}
                ) AS rollout
                RETURN rollout
            }}
            """,
            params={
                'firm_ext_id': firm_ext_id,
                'action_kind': action.kind,
                'params': action.parameters,
            },
        )
        
        # Filter rollout samples through hard constraints
        valid_samples = filter_through_locy_rules(
            outcome.rollout,
            modules=['bwm.accounting', 'bwm.regulation'],
            session=session,
        )
        
        asr = score_against_objective(valid_samples, objective)
        cod = counterfactual_distance(valid_samples, null_outcome.rollout)
        
        explanation = session.execute_locy(
            "EXPLAIN RULE pre_guidance_cut_warning "
            "WHERE f.ext_id = $firm_ext_id",
            params={'firm_ext_id': firm_ext_id},
        )
        
        analogs = session.execute_cypher(
            """
            MATCH (target:Firm {ext_id: $firm_ext_id})
            MATCH (analog:Firm)
            WHERE analog.ext_id <> target.ext_id
              AND similar_to(analog.firm_state_embedding,
                             target.firm_state_embedding) > 0.85
            RETURN analog.ext_id, analog.subsequent_outcome
            ORDER BY similar_to(...) DESC LIMIT 5
            """,
            params={'firm_ext_id': firm_ext_id},
        )
        
        results.append(ActionEvaluation(
            action=action,
            distribution=valid_samples,
            asr=asr,
            cod=cod,
            symbolic_trace=explanation,
            analogs=analogs,
        ))
    
    return rank(results, by=lambda r: r.asr - LAMBDA_RISK * r.risk())
```

The orchestration is thin. Most of the work happens inside the ASSUME blocks.

### 6.3 Audit and explanation contract

Every prediction surfaced to a user carries:

```
   ┌────────────────────────────────────────────────────────────┐
   │ Provenance triple                                          │
   │   • snapshot_id        (Uni snapshot pinned for the run)   │
   │   • model_versions     (resolved uni-xervo alias versions) │
   │   • module_versions    (Locy module semvers in effect)     │
   │                                                            │
   │ Explanation                                                │
   │   • Symbolic trace from EXPLAIN RULE — top contributing    │
   │     rules with their PROB contributions                    │
   │   • Graph-edge attributions — which neighbor states         │
   │     influenced this prediction                             │
   │   • Analog cases — top-K nearest historical firm-states    │
   │     via cross-encoder rerank (uni-xervo)                   │
   │   • Approximate-group flags — if MNOR/MPROD fell back to   │
   │     independence-assumption, this is surfaced              │
   │                                                            │
   │ Counterfactual                                             │
   │   • COD per candidate action                                │
   │   • ABDUCE results if user asks "what would change this?"  │
   │                                                            │
   │ Calibration                                                │
   │   • P10 / P50 / P90 of distribution                        │
   │   • Reliability score under most recent BWM-Bench          │
   │     calibration test for this regime / horizon              │
└────────────────────────────────────────────────────────────┘
```

This satisfies CR-1 (decision-centric explanation) and CR-3 (model risk documentation, SR 11-7-compliant). Single-number predictions without uncertainty and provenance are prohibited at the API surface.

### 6.4 ABDUCE as a first-class user query

Risk teams and strategy teams ask "what would have to change for X?" all the time. ABDUCE answers this directly.

```cypher
// What would need to change so this firm avoids a covenant breach?
ABDUCE NOT covenant_breach_4q
WHERE firm.ext_id = 'TARGET-CORP'
RETURN required_conditions
```

The ABDUCE runtime executes savepoint+mutate+re-evaluate+rollback per candidate modification. Results are minimum-change sets of modifications (e.g., "reduce debt by $200M and improve EBITDA margin by 150bps" or "extend maturity profile to push >25% maturity out beyond 2 years"). This becomes a Layer 5 user-facing query rather than internal optimization machinery.

### 6.5 What's not in v3

```
   Deferred to v4+ or future versions:
   ─────────────────────────────────────
   
   • Active information expansion. v3 does not autonomously
     design data-gathering interventions. L3-style active
     hypothesis testing waits for the Norn L3 spec.
   
   • Persistent model revision triggered by deployment
     evidence. v3 has periodic retraining only.
   
   • Federated multi-tenant evidence sharing. v3 keeps
     tenant data strictly isolated; pooled-evidence
     improvements come later.
   
   • Real-time (sub-day) prediction updates. v3 cadence
     is quarterly; intra-quarter events update state but
     not predictions.
```

---

## 7. Data pipeline

The pipeline is BWM's largest engineering surface and the source of most operational risk.

### 7.1 Source modalities

```
   Modality           Sources                            Cadence
   ──────────────────────────────────────────────────────────────
   Financials         SEC EDGAR (XBRL), S&P, equivalents Quarterly
   Filings text       EDGAR full-text + equivalents      Event
   Earnings calls     Transcript vendors                 Quarterly
   Market data        Polygon, exchange direct            Daily
   Macro              FRED, OECD, IMF, ECB               Monthly
   News               GDELT, vendor feeds                Daily
   Insider trades     EDGAR Form 4                       Event
   Patents            USPTO, EPO                         Event
   Hiring signals     Job postings, vendor LinkedIn      Monthly
   Supply chain       10-K disclosure + vendor feeds     Event
   Industry codes     NAICS / GICS authority             Annual
```

### 7.2 Point-in-time enforcement via BTIC + snapshots

Every ingested record is tagged with:

```
   Field              Type         Source
   ───────────────────────────────────────────────────────────
   valid_from         BTIC         When event/observation
                                   occurred (with epistemic
                                   certainty if uncertain)
   
   knowable_from      Timestamp    When this data became
                                   publicly knowable (filing
                                   date, press release time,
                                   measurement publication)
   
   recorded_at        Timestamp    When BWM ingested this
                                   record
   
   belief             Float        Confidence in the value;
                                   1.0 for definitive, <1.0
                                   for uncertain (e.g., news
                                   sentiment)
   
   perspective        String       Source identifier (used
                                   for multi-source merge)
   
   policy_tags        GSet         Residency, sensitivity,
                                   jurisdiction
   
   source_certainty   Float        Source reliability prior
                                   (e.g., 10-K = 1.0,
                                   news = 0.7)
```

Restatements are modeled as new records with `SUPERSEDES` edges to the records they correct, never by mutating existing records. This is the CPTE convention from Norn applied to financial data.

Query patterns enforce point-in-time semantics:

```cypher
// All facts knowable as of 2024-08-15
MATCH (filing:Filing) TIMESTAMP AS OF '2024-08-15'
WHERE filing.knowable_from <= datetime('2024-08-15')
  AND btic_contains_point(filing.period_covered,
                          datetime('2024-07-31'))
RETURN filing
```

The `TIMESTAMP AS OF` clause selects the Uni snapshot at that wall-clock time; the `knowable_from <=` filter enforces the additional constraint that the data was publicly available by that date (handles the case where data was ingested into BWM before being publicly knowable).

### 7.3 Entity reconciliation via UniId

The hard problem: a single firm appears across data sources with different identifiers (CIK on SEC, ticker on exchanges, LEI for regulatory, vendor-specific IDs for vendors X, Y, Z). Mergers, splits, and renames make this harder.

BWM's strategy:

```
   Stage                          Mechanism
   ──────────────────────────────────────────────────────────
   
   1. Stable identity             UniId (SHA3-256 content
                                  hash) computed from canonical
                                  attribute set: (label,
                                  legal_name_normalized,
                                  incorporation_jurisdiction,
                                  incorporation_date)
   
   2. Per-source ext_id           Each source's identifier
                                  attached to its source-specific
                                  label (Firm.ext_id = CIK;
                                  Listing.ext_id = ticker;
                                  separately stored vendor IDs)
   
   3. Reconciliation rules        Locy rules in bwm.identity
                                  (separate module) propose
                                  same-firm bindings; high-
                                  confidence bindings DERIVE
                                  IS_SAME_FIRM edges
   
   4. Merger / split tracking     Modeled explicitly via
                                  ACQUIRED_BY / SPUN_FROM
                                  edges with BTIC windows
```

Multi-source value conflicts (e.g., two vendors report different headcount for the same firm in the same quarter) use Uni's CRDT types: `LWWRegister` if last-writer-wins is acceptable, `VCRegister` if causal ordering matters, manual conflict resolution surfaced for audit when neither applies.

### 7.4 Graph construction

```
   Edge type        Source                              Cadence
   ──────────────────────────────────────────────────────────────
   SUPPLIES         10-K customer disclosure +          Quarterly
                    vendor relationship data
   
   COMPETES_WITH    SIC/NAICS/GICS overlap +            Quarterly
                    product-line text matching +
                    analyst report mining
   
   LENDS_TO         8-K material contract +             Event
                    syndicated loan databases
   
   HOLDS_EQUITY_IN  13F filings + cap-table data        Quarterly
   
   ROLE_AT          Form 4 + LinkedIn vendor data       Event
   
   BELONGS_TO       NAICS/GICS authority + override     Annual +
                    via expert reclassification         event
```

Graph data is the most incomplete and most expensive to maintain. Edge presence carries a `belief` property (often <1.0) and BWM's prediction surface explicitly reports graph coverage per prediction (how many of the expected neighbor edges were present in the queried snapshot).

### 7.5 Ingest pipeline architecture

Bulk-load path uses Uni's bulk-ingest API (per UR-5 latency target). Streaming updates use the regular write path with WAL durability.

```
   ┌────────────────────────────────────────────────────────────┐
   │                                                            │
   │  Source                                                    │
   │    │                                                       │
   │    ▼                                                       │
   │  Parse + normalize                                         │
   │    │  (form-type-specific code; XBRL processing for        │
   │    │   filings; text processing for earnings calls;        │
   │    │   etc.)                                               │
   │    ▼                                                       │
   │  Validate against Uni schema + PIT fields                  │
   │    │  (reject if missing valid_from, knowable_from,        │
   │    │   belief, perspective)                                │
   │    ▼                                                       │
   │  Compute embeddings (uni-xervo auto-embed for text;        │
   │  bwm/event-embedding model for events)                     │
   │    │                                                       │
   │    ▼                                                       │
   │  Bulk-load batch into Uni                                  │
   │    │  (uni-db bulk-ingest API, ~10K records/batch)         │
   │    ▼                                                       │
   │  Validate against bwm.accounting + bwm.regulation          │
   │  hard rules                                                │
   │    │  (Locy QUERY; violations flagged to human review,     │
   │    │   not rejected — restatements occasionally violate)   │
   │    ▼                                                       │
   │  Trigger affected rule re-evaluation                       │
   │    │  (Uni's semi-naive evaluator handles incremental      │
   │    │   updates efficiently)                                │
   │    ▼                                                       │
   │  Materialize ANALOG_OF edges via Locy DERIVE for           │
   │  newly-ingested firm states                                │
   │                                                            │
   └────────────────────────────────────────────────────────────┘
```

---

## 8. BWM-Bench

BWM-Bench is the evaluation harness for BWM. It is designed as a regime-specific instance of MREP (Minimal Reproducible Evaluation Package) per Norn conventions.

### 8.1 MREP conformance

```
   MREP component               BWM-Bench implementation
   ──────────────────────────────────────────────────────────
   
   1. Version locking           Pinned Uni snapshot +
                                pinned model versions +
                                pinned Locy module versions
                                (the provenance triple, § 4.4)
   
   2. Trace logs                Full prediction inputs,
                                intermediate Locy rule
                                derivations, model
                                invocations, outputs.
                                Stored for replay.
   
   3. Failure taxonomy          Automated classifier
                                mapping prediction failures
                                to the five Chu et al. L2
                                failure modes (compounding
                                error, state aliasing,
                                controllability failure,
                                exploitability, calibration
                                failure)
   
   4. Tail statistics           Stratified bootstrap CIs;
                                IQM; performance profiles
                                across firm characteristics
                                (size, industry, region,
                                regime period)
   
   5. Boundary condition        Explicit per-task declaration:
      mapping                   "this task tests long-horizon
                                coherence at H=8Q for the
                                financial-services subsector
                                under stable-rate macro
                                conditions"
```

### 8.2 Held-out evaluation periods

Entirely excluded from training. These are the regime-change events where calibration matters most.

```
   Window                       Significance
   ──────────────────────────────────────────────────────────
   2008-Q3 to 2009-Q2           Financial crisis (broad,
                                financial-services concentrated)
   
   2020-Q1 to 2020-Q3           COVID shock (sudden, global,
                                cross-sector)
   
   2022-Q1 to 2022-Q4           Rate hike cycle (regime change)
   
   2023-Q1 (SVB window)         Bank-specific cascade
                                (high-frequency, contagion)
   
   2022-Q1 to 2023-Q2           Energy crisis (sector-
                                specific, geopolitical)
```

For each window, BWM-Bench reports per-task ASR, COD, and CVR with stratified-bootstrap CIs.

### 8.3 Task categories

```
   ┌────────────────────────────────────────────────────────────┐
   │                                                            │
   │ Category A: Long-horizon coherence                         │
   │   • Regime-class prediction over H ∈ {1,2,4,8,12} quarters │
   │     reported as degradation curve, not single-horizon      │
   │     accuracy                                               │
   │   • Path-validity rate (fraction of rollout trajectories   │
   │     that maintain accounting identities + sign +           │
   │     conservation across all H steps)                       │
   │   • Cumulative state-distance from ground truth            │
   │                                                            │
   │ Category B: Intervention sensitivity                       │
   │   • COD(k) for each action_kind in the vocabulary,         │
   │     measured against historical-analog effect sizes        │
   │   • Action-controllability: changing the action at         │
   │     step k must produce a directionally correct shift      │
   │     in the predicted trajectory                            │
   │   • Action-magnitude sensitivity: doubling an action's     │
   │     parameter (e.g., acquisition value) should            │
   │     directionally amplify its predicted effect             │
   │                                                            │
   │ Category C: Constraint consistency                         │
   │   • Hard CVR (must be 0 by construction)                   │
   │   • Soft CVR distribution across PROB thresholds           │
   │   • Cross-regime consistency: rollouts must satisfy        │
   │     accounting + regulatory + supply-chain conservation    │
   │     jointly, not just one at a time                        │
   │                                                            │
   │ Category D: Calibration                                    │
   │   • Reliability diagrams on stable + crisis periods        │
   │   • Sharpness (P10-P90 spread / empirical spread)          │
   │   • OOD detection: does the model flag crisis periods      │
   │     as out-of-distribution before they happen?             │
   │                                                            │
   │ Category E: Failure taxonomy detection                     │
   │   • Each of the five Chu et al. L2 failure modes has        │
   │     a dedicated detector (per v2 § 11.4 thresholds)         │
   │   • Failures triggering any detector are surfaced in       │
   │     the BWM-Bench report                                   │
   │                                                            │
   │ Category F: Decision-task ASR                              │
   │   • Specific downstream tasks where action choice has      │
   │     ground-truth historical resolution: e.g., "M&A         │
   │     screening: out of these N proposed deals, which        │
   │     subsequently completed and produced positive 4Q        │
   │     outcomes for the acquirer?"                            │
   │   • Reported as ASR, not classification AUROC              │
   │                                                            │
   └────────────────────────────────────────────────────────────┘
```

### 8.4 Why this is decision-centric, not prediction-centric

Per v2 § 11.2 and Chu et al. § 6: prediction-accuracy metrics (AUROC on event classification, MSE on forecasted values) test L1 quality. Establishing L2 capability requires testing decision-usability under composition. BWM-Bench primary metrics are ASR / COD / CVR. Prediction-centric secondary metrics are reported for diagnostic purposes but cannot establish L2 capability claims.

### 8.5 Public benchmark commitment

BWM-Bench task definitions, held-out window specifications, and the failure-taxonomy detector thresholds are published openly. Specific snapshot contents (the Uni snapshot pinning the data) are versioned. This is consistent with the Norn MREP standard and gives external researchers a reproducible target for cross-regime comparison.

---

## 9. Phases and acceptance criteria

Each phase declares which L-level capability is being established and what gate must clear before moving on. Phase G is explicitly per-tenant adaptation, not L3. Phase H is future, conditional on Norn L3 spec progress.

### 9.1 Phase A: Data foundation

```
   Output      Production-grade ingest pipeline; snapshotted
               foundation corpus of ~50K entities × 25 years
               × all modalities
   
   Acceptance  • All records carry valid_from (BTIC),
                 knowable_from, belief, perspective,
                 policy_tags, source_certainty
               • PIT spot-checks pass on 50 historical events
                 across sectors and crises
               • Foundation Uni snapshot created and named
                 (BWM-FOUNDATION-v1)
               • Hard CVR on bwm.accounting + bwm.regulation
                 ≤ 2% (allowance for known restatement edge
                 cases; >2% means data error)
```

### 9.2 Phase B: SSM + JEPA + inverse-dynamics pretraining

```
   Output      Trained bwm/firm-state-encoder/v1,
               bwm/forward-dynamics-jepa/v1, and
               bwm/inverse-dynamics/v1; registered with
               uni-xervo
   
   Acceptance  • Regime-class probe top-1 ≥ 70% on stable
                 periods
               • Embedding-space silhouette score ≥ 0.5
                 across firm cohorts
               • Inverse-dynamics top-3 action prediction
                 ≥ 60% on labeled action set
               • Models exported to ONNX and registered
                 with uni-xervo successfully
               • Locy rule using classify(...) on a sample
                 BWM model returns expected results in
                 integration test
               • Architecture sweep complete: SSM:attention
                 ratio ∈ {pure, 7:1, 3:1, 1:1} and model size
                 ∈ {150M, 350M, 700M} evaluated on train/val
                 gap; selected config documented in ADR-1
   
   Capability  L1 (state inference + forward dynamics
   level       + inverse dynamics + observation decoding)
```

### 9.3 Phase C: Latent decoders

```
   Output      Trained bwm/latent-decoder/v1 family —
               revenue, margin, leverage, growth,
               working-capital regime decoders
   
   Acceptance  • All Phase B criteria continue to hold
               • Per-regime classification accuracy ≥ 75%
                 across stable periods
               • Distributional decoder output passes
                 calibration smoke test (P50 ≈ median of
                 truth on training tail)
   
   Capability  L1 full (all four operators)
   level
```

### 9.4 Phase D: Diffusion / flow head

```
   Output      Trained bwm/diffusion-head/v1; complete
               distributional prediction surface across
               H ∈ {1,2,4,8,12}
   
   Acceptance  • All Phase C criteria continue to hold
               • Reliability diagram on stable regimes:
                 90% CI contains 88-92% of actuals
               • Reliability diagram on held-out crises:
                 90% CI contains ≥ 75% of actuals
               • Sharpness within 1.5× empirical spread
                 on stable regimes
               • 4-step inference latency ≤ 100ms per
                 sample on representative hardware
   
   Capability  L1 + distributional substrate
   level       (enables COD computation; pre-L2)
```

### 9.5 Phase E: Rule library + uni-xervo integration

```
   Output      bwm.accounting, bwm.regulation,
               bwm.empirics (initial ~500 rules),
               bwm.strategy, bwm.supplychain Locy modules
   
   Acceptance  • All Phase D criteria continue to hold
               • Hard CVR = 0% on rollout samples (rejection
                 sampling working)
               • Soft-rule explanation traces rated ≥ 80%
                 useful by domain-expert audit (n=100
                 samples across sectors)
               • Inverse-dynamics restatement detection
                 AUROC ≥ 0.65 (neural head) AND ABDUCE
                 produces matching explanation on ≥ 60%
                 of true positives
               • Locy module evaluation latency at full
                 corpus scale ≤ 60s (UR-1 validation)
   
   Capability  L1 full + governing-law enforcement +
   level       neuro-symbolic composition working
```

### 9.6 Phase F: L2 elevation training

This is the gating phase for the L2 capability claim. Joint training (with frozen earlier layers) explicitly targets the three boundary conditions.

```
   Output      L2 Simulator passing the boundary
               conditions on BWM-Bench
   
   Acceptance  • All Phase E criteria continue to hold
               • Long-horizon coherence: degradation
                 curve no steeper than baseline + 2σ on
                 8Q horizon; rollouts decision-usable
                 through 8Q
               • Intervention sensitivity: COD(k) > 0
                 with directionally-correct sign on
                 ≥ 90% of action_kind × horizon cells
               • Constraint consistency: soft CVR < 5%
                 on returned samples
               • All 5 failure-mode detectors clear on
                 BWM-Bench held-out windows
               • ASR on Category-F tasks ≥ baseline
                 by 15% on stable periods, ≥ baseline
                 by 5% on crisis periods
   
   Capability  L2 (full)
   level
```

### 9.7 Phase G: Per-tenant adaptation

```
   Output      Per-customer LoRA adapters on relevant
               BWM models; per-tenant Locy modules
               composing bwm.* base modules with
               customer overrides
   
   Acceptance  • All Phase F criteria continue to hold
               • Per-tenant adaptation lifts ASR by ≥ 10%
                 over foundation baseline on
                 customer-specific tasks
               • Strict tenant isolation: no cross-tenant
                 gradient flow; no cross-tenant rule
                 leakage (verified via canary tests)
               • Customer-specific rules can be added,
                 modified, or removed without retraining
                 foundation models
   
   Capability  L2 + per-tenant adaptation
   level       (explicitly NOT L3)
```

### 9.8 Phase H: Evidence-driven model revision (future)

Conditional on Norn L3 specification maturing to the point of providing concrete conformance criteria for evidence-driven revision.

```
   Output      Partial L3 Evolver: deployment evidence
               triggers regression-gated model + rule
               updates
   
   Acceptance  • Evidence-grounded diagnosis: prediction
                 failures attributed to specific
                 components with replayable evidence
               • Persistent asset updates: fixes promoted
                 as new versioned model aliases and Locy
                 rule modules, not in-context patches
               • Governed validation: regression gates
                 + canary deployment + rollback policy
                 operational
               • Cross-cycle improvement: ASR improves
                 across revision cycles k without
                 regressing on held-out probes
   
   Capability  Partial L3 (no autonomous data
   level       collection; reactive revision only)
```

---

## 10. Open questions

Questions that the spec assumes but has not validated. Each is tagged with where it lives (BWM internal, BWM/Uni boundary, BWM/Norn boundary).

```
   ┌────────────────────────────────────────────────────────────┐
   │ OQ-1   [BWM/Uni]                                           │
   │ ─────────────────                                          │
   │ UR-1 verification: at what entity / rule count does the    │
   │ Locy stratified evaluator slow down meaningfully? Spec     │
   │ assumes ≤ 60s for full bwm.* evaluation on representative  │
   │ BWM corpus. Will need a microbenchmark in Phase E.         │
   │                                                            │
   │ ────────────────────────────────────────────────────────── │
   │ OQ-2   [BWM/Uni]                                           │
   │ ─────────────────                                          │
   │ UR-2: established pattern for projecting BTIC values to    │
   │ numeric features at the SSM encoder boundary. Likely BWM   │
   │ implements a small helper (start_offset_quarters,          │
   │ duration_quarters, certainty_onehot, granularity_onehot)   │
   │ but want to confirm no Uni-side pattern exists already.    │
   │                                                            │
   │ ────────────────────────────────────────────────────────── │
   │ OQ-3   [BWM/Uni]                                           │
   │ ─────────────────                                          │
   │ UR-3: empirical cost model for exact_probability on        │
   │ recursive supply-chain rules. BWM default is               │
   │ exact_probability=false with opt-in for specific stress    │
   │ tests, but the BDD-variable threshold at which fallback    │
   │ becomes routine needs measurement.                         │
   │                                                            │
   │ ────────────────────────────────────────────────────────── │
   │ OQ-4   [BWM internal]                                      │
   │ ─────────────────────                                      │
   │ Inverse-dynamics ground truth: logged action data is       │
   │ sparse and biased toward observable corporate actions.     │
   │ Should we use LLM-extracted action labels from filing      │
   │ text as additional weak supervision, accepting the noise?  │
   │                                                            │
   │ ────────────────────────────────────────────────────────── │
   │ OQ-5   [BWM internal]                                      │
   │ ─────────────────────                                      │
   │ Counterfactual identification: pure observational COD is   │
   │ unreliable because we cannot observe untaken actions. Do   │
   │ we model treatment assignment explicitly (propensity        │
   │ scores, instrumental variables in macro shocks), or rely   │
   │ on analog retrieval to surface uncertainty in              │
   │ counterfactual claims? Chu et al.'s "intervention          │
   │ sensitivity" assumes the former; current spec is closer    │
   │ to the latter.                                             │
   │                                                            │
   │ ────────────────────────────────────────────────────────── │
   │ OQ-6   [BWM internal]                                      │
   │ ─────────────────────                                      │
   │ Long-horizon rollout strategy past 12Q: autoregress        │
   │ (compounding-error risk) or train direct long-horizon      │
   │ predictors (data sparsity at H ≥ 20)? Likely answer:       │
   │ direct predictors out to some horizon then autoregressive  │
   │ with explicit uncertainty inflation.                        │
   │                                                            │
   │ ────────────────────────────────────────────────────────── │
   │ OQ-7   [BWM/Norn]                                          │
   │ ──────────────────                                         │
   │ When does Phase H become tractable? Specifically, what     │
   │ shape of Norn L3 spec gives BWM concrete conformance       │
   │ criteria for evidence-driven revision? This is a question  │
   │ for the Norn maintainers; BWM doesn't drive it.            │
   │                                                            │
   │ ────────────────────────────────────────────────────────── │
   │ OQ-8   [BWM/Norn]                                          │
   │ ──────────────────                                         │
   │ Several BWM patterns might generalize to a Norn            │
   │ "business-regime conformance profile" (event_kind          │
   │ registry conventions for filings + corporate actions,      │
   │ BTIC patterns for restatement-prone data, MNOR-based       │
   │ probabilistic propagation through dependency graphs).      │
   │ Worth proposing to Norn maintainers once Phase E ships.    │
   │                                                            │
   │ ────────────────────────────────────────────────────────── │
   │ OQ-9   [BWM/Norn]                                          │
   │ ──────────────────                                         │
   │ Federated multi-tenant evidence sharing: Norn position     │
   │ paper claims federation; Uni doesn't yet implement it.     │
   │ For BWM, this is the long-term answer to "how do           │
   │ customers benefit from each other's data without           │
   │ violating isolation." Not in v3 scope, but worth tracking. │
   │                                                            │
   └────────────────────────────────────────────────────────────┘
```

---

## 11. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Point-in-time errors invalidate all backtests | Critical | Aggressive PIT engineering in Phase A; spot-check 50+ historical events; third-party audit before Phase B |
| Phase F destabilizes pretrained representations | High | Frozen Layer 1-2-3 by default; light Layer 2 fine-tune only; rollback on Phase B-E regression |
| Soft-rule curation cost is unsustainable | High | LLM-assisted drafting + expert review; rule versioning; small initial set (~500), grown only as evidence accrues |
| Graph data is too incomplete for supply-chain rules to be reliable | High | Edge-presence belief on every edge; graceful degradation; "graph coverage" metric reported per prediction |
| Locy rule evaluation doesn't scale to full corpus | Medium | UR-1 microbenchmark in Phase E; if it fails, work with Uni team on partitioned evaluation |
| Confusing Phase G with L3 in marketing or product | Medium | Explicit Phase G ≠ L3 in this spec; Phase H clearly future-tagged |
| Customer fine-tuning leaks signal across tenants | Critical | Strict per-tenant adapter weights; no shared gradients; canary tests in Phase G acceptance |
| OOD performance collapses in regime change | Medium | Train on multiple historical crises; explicit OOD calibration tests; flag-rather-than-predict on suspicious inputs |
| Compute exceeds projections | Medium | Phased training; Phase B is the only large run; later phases cheaper |
| Regulator rejects opacity of latent layer | Medium | EXPLAIN RULE + analog retrieval + symbolic trace at every prediction; Lakatosian framing positions Locy as the auditable substrate |
| BWM-Bench becomes a target rather than diagnostic | Medium | Snapshot version locking; rotate held-out crisis windows; publish failure taxonomy openly |
| Uni feature gap blocks a phase | Medium | UR-1 through UR-8 validation early; engage with Uni team via feature requests, not internal forks |

---

## 12. Cross-references and external dependencies

```
   Document                              Status / version
   ──────────────────────────────────────────────────────────
   
   Chu et al. (2026) — Agentic World     Reference (cited)
   Modeling: Foundations, Capabilities,
   Laws, and Beyond
   
   Norn — Become, Being, and Shall Be    Conventions consumed
   (vision paper)                        (BWM is a conformant
                                         application)
   
   Norn — Foam / Weave / Perceive        CPTE-style metadata
   1.0+ Requirement Specification        conventions for
                                         event-type labels
   
   Norn — MREP standard                   BWM-Bench is an
   (per Chu et al. § 6)                  MREP instance
   
   Uni Black Book                         Substrate API
   (rustic-ai/uni-db/docs/                reference
   UNI_BLACK_BOOK.md)
   
   Uni-Xervo documentation                Model runtime
                                         reference
   
   BWM v1 spec (this document's          Superseded
   predecessor)                          
   
   BWM v2 spec                            Superseded by v3
                                         (positioning and
                                         decomposition);
                                         analysis preserved
   
   BWM training plan                      Separate document
   (compute, schedule, dataset            (deferred)
   sourcing)
   
   BWM-Bench task catalog                Appendix C
   (this spec)
```

---

## Appendix A: BWM event_kind registry (initial)

The full registered set with shape constraints. Each `event_kind` has a registered shape (JSON-Schema), a default `belief` prior, a `policy_tags` floor, and a set of associated soft-rule contexts in `bwm.empirics`.

(Detailed schemas elided in this spec — to be authored as a separate machine-readable artifact in `bwm/schemas/event_kinds/`.)

Categories registered initially: filings (6), corporate actions (13), regulatory events (5), macro events (3), restatements (2), earnings events (3), insider transactions (3), news events (1 with sub-classification).

Total: ~36 registered event_kinds at Phase A; expected to grow to ~80 by Phase D as new categories are added.

---

## Appendix B: BWM Locy module library (overview)

The five modules with their rule-count targets per phase.

```
   Module                  Phase E    Phase F    Phase G
   ──────────────────────────────────────────────────────
   bwm.accounting          ~80        ~100       ~120
   bwm.regulation          ~40        ~60        ~100
                                                  (per-tenant
                                                   jurisdictions)
   bwm.empirics            ~500       ~1500      ~2000
   bwm.strategy            ~100       ~300       ~400
   bwm.supplychain         ~100       ~300       ~400
                          ─────      ─────      ─────
   Total                   ~820      ~2260      ~3020
```

Each module is published as a versioned Locy source file in `bwm/locy/`. Per-tenant modules live in customer-specific repositories under tenancy isolation; BWM publishes the base modules and the composition pattern but not customer rules.

---

## Appendix C: BWM-Bench task catalog (overview)

Authored as a separate document; this is the index.

```
   Task family                    Count    Categories
   ──────────────────────────────────────────────────────────
   Regime-class prediction        15       A (long-horizon)
   (one per regime × 3 horizons)
   
   Action-COD                     45       B (intervention)
   (15 actions × 3 horizons)
   
   Cross-regime joint             20       C (constraint
   validity                                 consistency)
   
   Reliability / calibration      10       D (calibration)
   
   Failure-mode detection         15       E (failure
   probes                                  taxonomy)
   
   M&A screening                  4        F (decision ASR)
   Capital action screening       4        F
   Restatement detection          4        F
   Distress prediction            4        F
   Regulatory exposure            4        F
                                  ───
   Total                          125
```

Detailed task definitions, success criteria, and bootstrap protocol live in `bwm-bench/tasks/`.

---

*End of specification v3.*
