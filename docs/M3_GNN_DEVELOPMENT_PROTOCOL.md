# M3 — independent GNN development

Filed 2026-09-07, before any GNN is fit under the corrected context. The
machine-readable declaration is `configs/m3_gnn_development.yaml`; this document
is the argument behind it. Status:
`M3_DECLARED_RECONNAISSANCE_COMPLETE_NO_FIT_AUTHORISED`.

**This phase-step trains nothing.** It records what the historical GNN was,
audits it against the fairness contract, proposes two candidates, derives the
screen datasets from graph-only evidence, freezes the selection rule, estimates
the compute, and stops.

## 1. Why M3 is its own phase

M2B closed semantic selection at S3 and QLS-v2 is frozen at 3,585 parameters.
The risk that governed M1A through M2B was over-tuning QLS. That risk is now
retired, and a different one replaces it: **tuning the GNN against QLS**.

The mechanism is subtle enough to be worth naming precisely. Suppose the GNN
screen ran with QLS's numbers visible. Every decision — which operator, which
depth, whether to add a width variant, when to stop screening — would then be
made by someone who could see which choices narrowed the gap. Nothing dishonest
need happen. The screen would simply stop being a search for the best GNN and
become a search for the GNN that loses least, and the M4 comparison would be
between QLS and an object selected for its relationship to QLS. That comparison
would have no meaning, and no amount of care in M4 could restore it.

So the ordering is enforced structurally rather than by intention:

```
M2 / M2B outcomes  ->  QLS FROZEN
M3 (GNN candidates only, no QLS number consulted)  ->  GNN FROZEN
M4  ->  QLS vs GNN
```

M3 answers **"which GNN architecture is best among the preregistered GNN
candidates?"** It does not answer, and must not be read as answering, "which GNN
architecture beats QLS?" The selection rule in §6 contains no QLS term; the
screen datasets in §5 were derived from evidence that predates every QLS number.

## 2. Historical GNN archaeology

Recovered from source and **verified by live instantiation**, not from prose.
Where a document and the code disagreed, the code won; where a parameter count
could be measured by constructing the module, it was measured.

| item | value | source |
|---|---|---|
| class | `SeedAwareMessagePassingOperator` | `src/mp_retrieval/operator_models.py:227` |
| base | `MessagePassingOperator` → `OperatorModel` | `operator_models.py:158` |
| factory | `build_seed_aware_message_passing` | `scripts/run_sa_mlp_confirmation.py:441` |
| operator | **per dataset** — see §2.1 | `configs/candidate_budget.yaml:27-54` |
| depth | 1 | `configs/candidate_budget.yaml:62` |
| hidden width | 64 | CLI default |
| node input width | 1537 = 1536 embedding + 1 seed scalar | `Linear(embedding_dim + 1, hidden_dim, bias=False)` |
| query conditioning | `Linear(1536,64)` then a residual head `Linear(64,64)→GELU→Dropout→Linear(64,64)`; query never message-passes | `MessagePassingOperator.forward_seed_aware` |
| seed awareness | one binary retrieval-seed scalar concatenated before projection — costs exactly 64 parameters | `forward_seed_aware` |
| layer update | `h ← LayerNorm(h + Dropout(GELU(conv(h, edge_index))))` | `forward_seed_aware` |
| activation / dropout | GELU / 0.2 on the update | |
| readout | `dot(normalize(h), target[batch]) / 0.07` | `OperatorModel.similarities` |
| loss | listwise soft cross-entropy, uniform mass over in-pool golds | `scripts/run_operator_screen.py:135` |
| optimiser | AdamW, lr 1e-3, wd 1e-4, 3 epochs, batch 16 | `run_sa_mlp_confirmation.py:368` |
| graph | `G[Cq]` — `dataset.induced_subgraph(query)` indexes by `query.candidate_index` only | `src/mp_retrieval/complete_data.py:92` |

Measured parameter counts at E=1536, H=64:

| depth | gcn | sage | gat | gin |
|---|---:|---:|---:|---:|
| 1 | 209,280 | 213,376 | 213,568 | 213,440 |
| 2 | 213,568 | 221,760 | 222,144 | 221,888 |

The distribution matters more than the totals. The two 1536-wide input
projections are 196,672 parameters and the query head another 8,320 — **204,992
before a single message-passing parameter**. One GCN layer adds 4,288. Depth is
nearly free; the projections *are* the model. That is why §4 proposes a depth
variant and not a width variant.

Three findings from this archaeology change the plan as originally sketched, and
each is recorded because discovering it later would be worse than declaring it
now.

### 2.1 There is no single historical operator

`configs/candidate_budget.yaml` froze a **different operator family per
dataset**, selected by an earlier operator screen:

| dataset | operator |
|---|---|
| 2wiki_clean, webqsp, metaqa | gat |
| musique_clean, squad_clean | gcn |
| hotpotqa_clean | gin |

"The historical operator" is therefore not well-defined across datasets. It
happens that both datasets the graph-only rule selects in §5 were frozen at
`gat`, so a single anchor exists *for this screen* — but that is an output of
the derivation, not an input to it, and it would not have survived a different
one. A screen containing hotpotqa_clean would have faced `gin` against `gat`
with no principled way to call either "the" historical operator, and would have
needed a different candidate definition.

### 2.2 G0 is not a byte-identical reproduction of the historical model

The historical GNN received node embeddings, the query embedding, one binary
seed indicator, and `edge_index`. It received **no retrieval features at all**.

QLS-v2's `BASE` block is four columns — `seed_identity`, `dense_rr`,
`splade_rr`, `agreement` (`src/mp_retrieval/m1a_screen.py:54`) — and the
historical GNN had only the first. Under the fairness contract the M3 GNN must
receive all four, plus `NODE_ROLE`, which takes its node input width from
**1537 to 1541** — +4 columns and +256 parameters in the node projection, of
which 1,540 are informative under R1 and R2.

`NODE_ROLE` is carried in every regime and is identically zero under R1 and R2,
because that is exactly what QLS-v2 does: it keeps the column in
`MASTER_COLUMNS` throughout and zeroes it where nothing is admitted. A schema
that changed shape between regimes on one side but not the other would make the
two systems' inputs differ for a reason unrelated to either mechanism.

So G0 is the historical *operator, depth and width* with the input privilege
corrected — not the historical model re-run. Calling it "the historical
architecture" while quietly feeding it three new inputs would misdescribe both
the baseline and any gap it closes. It is the right thing to do (the contract
requires it) and the wrong thing to leave unsaid.

### 2.3 The historical defect sat exactly at the historical depth

`G[Cq]` keeps an edge only when both endpoints are candidates, so radius 1 is
precisely where it deletes most. On the sealed graph, for the two screen
datasets:

| | 2wiki_clean | metaqa |
|---|---:|---:|
| median node-pooled radius-1 retention | 0.111 | 0.154 |
| boundary cut | 0.839 | 0.904 |
| candidates isolated inside `G[Cq]` | 37.5% | 41.2% |

A one-layer GNN on that substrate is induction-starved at its own receptive
field. Historical results are **not rewritten** — they stand as measured, under
the context they were measured in. M3 is a new corrected-context track and says
so.

## 3. Corrected graph privilege

The regimes are already frozen and are reused unchanged. M3 defines no new
context and rebuilds no features.

| regime | scored | context | radius supplied |
|---|---|---|---|
| R1 | `Cq` | `Cq` | 0 beyond `Cq` |
| R2 | `Cq` | `U2 = TARGET_H1(Cq) = Cq ∪ N1_in(Cq)` | 1 |
| R3_BOUNDED | `C3 = Cq ∪ A64` | `U3 = U2 ∪ A64` | 1 plus A64 admission |

`R3_FULL_REEXPANSION` remains prohibited. A64 is frozen.

### 3.1 Depth and radius are not independent — and this bites

§9.2 of the fairness contract is explicit: an L-layer GNN has an L-hop receptive
field, `TARGET_H1` must not be declared the graph context for all depths, and
any truncation is a **limitation of the condition**, reported in the same table
as the result it affects and never counted as evidence against the baseline.

Applying that clause to the frozen regimes:

| | depth 1 | depth 2 |
|---|---|---|
| R1 | **truncated** (this is the historical defect) | truncated |
| R2 | full radius-1 field | truncated |
| R3 | full over `U3` | truncated |

**There is no un-truncated condition for depth 2 among the frozen regimes.** The
G0-vs-G1 contrast is therefore partly depth against truncation and cannot be
read as a clean depth effect. That is stated here, before any fit, and is
required in every table that reports it.

Supplying `TARGET_H2` would fix it and is deliberately **not** proposed: a new
context means new feature stores, a new privilege for both systems, and a
re-derivation of the fairness argument — a larger intervention than this phase
is scoped for, and one that should be argued for on its own rather than smuggled
in as a screen detail.

What the screen *does* supply cleanly is **G0 under R1 against G0 under R2**:
same architecture, same depth, same inputs, and the only thing that changes is
whether the radius-1 context that one layer needs is actually present. That is
the corrected-privilege measurement, and it is un-truncated on the R2 side.

## 4. Information fairness

`docs/GRAPH_INFORMATION_FAIRNESS_CONTRACT.md` was filed before any comparison
and is binding as written. M3 adds no clause and relaxes none. What follows
records how M3 satisfies it and where it bites.

**Equal privilege, not equal representation.** Both systems receive the same
query, the same scored set, the same context nodes and induced edges from one
code path, the same frozen graph and node embeddings, seed identity, dense RR,
SPLADE RR, agreement, `NODE_ROLE` where the regime defines it, and the same
supervision, splits, loss family and seed protocol.

**The GNN does not receive SUPPORT or PATH.** This is correct and is preserved.
SUPPORT and PATH are QLS's fixed graph-summary *mechanism*, not extra
information: both systems get the same underlying graph, QLS reduces it to fixed
query-conditioned descriptors, the GNN learns aggregation over it. Handing the
GNN the descriptors as well changes the question from *"fixed summaries or
learned propagation?"* to *"does propagation add anything once the summaries are
supplied?"* — a different and much narrower experiment. GNN-plus-explicit-
summaries is a legitimate later control and is **not authorised in M3**; it is
named here so that if it is ever run it is recognisable as the separate
experiment it is.

Symmetrically, QLS receives nothing GNN-derived. No teacher, no distillation, no
logits as targets, no hidden representations, no GNN-selected features, no
gold-derived graph inputs on either side. QLS is frozen at M2B in any case, so
there is nothing left in M3 to contaminate.

### 4.1 The tuning-budget asymmetry this screen creates

§5 of the contract says the GNN gets "a tuning budget at least equal to
QLS-v2's… and the grids are reported." Honesty requires reporting that this
screen does not obviously clear that bar.

QLS-v2's architecture was selected across three phases: M1A screened feature
families, M2 selected the arm, M2B screened three semantic rungs over 42
evaluations. The GNN screen proposed here is **2 candidates over 6 cells**. On
any count of fits or of decisions made, that is smaller.

It is proposed anyway for two reasons. A Cartesian sweep over operators, depths
and widths is the failure this entire programme has been organised to avoid.
And — less obviously — a wide sweep would damage the firewall in practice: the
more GNN variants are screened, the more "the best GNN" becomes a best-of-many
statistic, which is optimistically biased exactly in the direction that would
flatter the baseline in M4. A small, audit-anchored candidate set is both
cheaper and cleaner.

The asymmetry is declared rather than hidden, and both grids are reported side
by side whenever an M4 comparison is written. **If review judges the budget
inadequate, the remedy is to widen the GNN screen in a dated amendment before
any fit — never afterwards because the first result was unwelcome.** And "the
GNN has 58× more parameters" is not a tuning budget; the contract forbids using
parameter count as a handicap in either direction.

## 5. Representative screen, derived from graph-only evidence

The requirement is at least one passage-style and one KB-style graph, selected
**mechanically from pre-QLS evidence**. Every quantity below comes from the
graph substrate audit and the graph-context D0 series, both of which predate
QLS-v2. No QLS effectiveness number entered this derivation.

| dataset | style | isolated frac. | boundary cut | median retention | isolated-gold queries | validation queries |
|---|---|---:|---:|---:|---:|---:|
| 2wiki_clean | passage | 0.375 | 0.839 | 0.111 | 20.0% | 3,000 |
| hotpotqa_clean | passage | 0.355 | 0.890 | 0.077 | 6.0% | 19,570 |
| musique_clean | passage | 0.224 | 0.792 | 0.200 | 10.3% | 3,987 |
| squad_clean | passage | 0.175 | 0.881 | 0.080 | 1.0% | 26,063 |
| metaqa | KB | 0.412 | 0.904 | 0.154 | 0.0% | 39,138 |
| webqsp | KB | 0.190 | 0.700 | 0.250 | 1.5% | 315 |

**Passage-style → `2wiki_clean`.** Highest isolated fraction of the four passage
graphs, highest isolated-gold prevalence of all six (so the repair has the most
to act on), and the smallest context of the four at 464.1 mean directed edges
against squad's 3,190.8. Damage, headroom and cost all point the same way, so no
trade-off has to be adjudicated.

**KB-style → `metaqa`.** The two KB graphs genuinely conflict, and the tie is
broken on panel size before graph properties are even reached. webqsp has **315
validation queries**; M2B held out 63 of them, where a single query moves recall
by 1.587pp. The rule in §6 uses a 0.50pp per-cell tolerance. A panel on which
one query is worth three times the tolerance cannot decide a per-cell clause, so
webqsp is unusable for this selection regardless of its graph — and it is also
the *least* damaged graph of the six (boundary cut 0.700, the lowest), so it
would be the weakest KB test even with a usable panel. metaqa is the most
graph-damaged dataset in the set on both isolation (0.412) and boundary cut
(0.904).

**The limitation metaqa carries, filed in advance.** metaqa has **0.0%** of
queries with an isolated gold — 0 of 195 measured in D0. Whatever the corrected
context does for isolated candidates, on metaqa it has almost nothing to act on
*for ranking*. A null result there is therefore weak evidence about the
correction, though it remains informative about architecture, since depth is a
receptive-field question that does not require isolated golds to be exercised.
The wrong reading — "the correction does not help on KB graphs" — is
pre-registered as wrong here so that it cannot be chosen afterwards. The right
reading is "metaqa cannot distinguish that."

**Cells:** `2wiki_clean × {R1, R2, R3}` and `metaqa × {R1, R2, R3}` — 6 cells,
12 logical fits across the 2 candidates.

Both datasets are also M2B cells, which conveniently means frozen candidate
pools and splits can be reused. That is a consequence of the derivation, not an
input to it. Neither was chosen because of how QLS performed there; QLS's
results on these datasets are not consulted anywhere in this phase.

## 6. The GNN-only selection rule

Frozen now, before any GNN number exists.

- **Primary metric:** recall@5.
- **Anchoring:** `SYMMETRIC_BEST_ANCHORED` — every candidate is measured against
  **the best GNN candidate** in each scope. Never against QLS, never against the
  historical GNN.
- **Scores:** `cell_score` = R@5 for one (dataset, regime); `dataset_score` =
  equal-weight mean over that dataset's declared regimes; `macro_score` =
  equal-weight mean over the screen datasets.
- **Admissible iff:** macro within **0.25pp** of the best GNN, *and* every
  dataset within **0.50pp**, *and* every cell within **0.50pp**.
- **Thresholds are not adjustable.** Carried from M2B with the same force, and
  they are the same values M2B used, so they were not chosen for this phase's
  data either.
- **Among admissible:** `LEXICOGRAPHIC_SYSTEMS_PARETO` — lower total uncached
  p95, then fewer parameters, then lower peak memory, then the filed
  deterministic rule (fewer message-passing layers, then earliest of G0/G1).
- **If none survives:** report `GNN_PARETO_CONFLICT` and stop for review,
  exactly as M2B did. Do not widen a tolerance, drop a clause, or reweight.

No QLS quantity appears anywhere in this rule — not as a threshold, not as an
anchor, not as a tie-break. This is checked by test, not only by inspection.

**Seeds:** seed 0 only. Five-seed confirmation remains prohibited. If
architecture selection is genuinely ambiguous afterwards, the smallest
sufficient three-seed resolution may be *proposed*, naming the specific cells it
would resolve — using M2B's template, including its requirement that the
resolution rule be frozen before the fits it judges.

## 7. Systems

The primary systems metric is **total uncached inference p50/p95/p99**, and the
span it covers is the whole cold path:

```
query arrives
  -> candidate and context resolution
  -> context edge gathering
  -> node and query input construction
  -> message passing
  -> candidate readout and scoring
```

**GNN-forward-only timing must not be reported as cold-query latency.**
Benchmarking the GNN from already-materialised edge tensors while charging QLS
for building its features would be the systems mirror of exactly the substrate
error the fairness contract exists to prevent. Components are persisted
separately — `context_materialization_ms`, `edge_gather_ms`,
`message_passing_ms`, `readout_ms`, `total_uncached_ms` — because context
construction is paid by both systems and cancels in a like-for-like comparison
(contract §6), so it has to be visible rather than folded in; recording the
components also makes the total auditable instead of asserted. Cached forward
latency is a secondary diagnostic.

## 8. Persistence

Per fit: checkpoint (reloading `strict=True`), per-query rows with the aggregate
**rebuilt from them** and matching exactly, query/candidate/context IDs, edge
fingerprint, graph/input contract hash, model-config hash, source commit,
trainable parameter count, training seconds, peak training VRAM, peak inference
VRAM and RSS, and the timing set above.

The contract-hash design is reused from `scripts/feature_build_contract.py`:
**hash the scientific inputs, not the declaration prose.** This file is mutable
by design — it will carry amendments — and making prose part of tensor identity
would either force spurious rebuilds or, worse, teach everyone to avoid editing
the declaration.

## 9. Workload and compute

2 candidates × 6 cells × 1 seed = **12 fits** in 2 containers (one per dataset,
so both candidates share a clock). No features are rebuilt and no new data is
sealed; M2's frozen cell masters, candidate pools, splits and node embeddings
are reused.

The cost basis is M2B's **measured** S4 fits on these six cells, totalling
**462.07 s**. S4 carries 205,217 parameters against the GNN's 213,568, so it is
a close parameter-matched proxy for everything except the message passing
itself.

| | R1 | R2 | R3 |
|---|---:|---:|---:|
| 2wiki_clean | 15.25 | 15.33 | 15.48 |
| metaqa | 106.12 | 121.06 | 188.84 |

Applying **provisional** multipliers of 1.5× (G0) and 2.0× (G1) gives 1,617.2 s,
2,425.8 s with 1.5× headroom, ≈ **$1.83 all-in** at $2.241/GPU-hour with
inference benchmarking and container overhead.

**The multipliers are guesses and are labelled as such.** They are guesses at
what message passing and context materialisation cost on top of a
parameter-matched forward pass. M2B is the reason for the label: its own
pre-launch estimate used a guessed 2.5× for S4 and the smoke measured something
different. A smoke must replace these with a measurement before any fan-out. The
proposed $2.50 ceiling is therefore **proposed, not filed** — the amendment that
authorises fitting must refile it against the smoke, exactly as M2B refiled 6.00
to 3.5732. The ledger line is `m3_gnn_development`, separate from every M2B line.

## 10. Gates

| gate | state |
|---|---|
| archaeology recorded (live instantiation) | ✅ earned 2026-09-07 |
| fairness contract bound | ✅ earned 2026-09-07 |
| screen derived from graph-only evidence | ✅ earned 2026-09-07 |
| selection rule frozen, no QLS term | ✅ earned 2026-09-07 |
| candidate set declared | ✅ earned 2026-09-07 |
| depth-2 truncation declared | ✅ earned 2026-09-07 |
| runner exists | ❌ |
| instrumentation tests pass | ❌ |
| smoke measures a real GNN fit | ❌ |
| compute ceiling refiled against measurement | ❌ |

A gate is turned true by doing the thing it names and pointing at the artifact
that proves it, never by deciding it does not apply.

## 11. Stop

This phase-step produced the declaration, the archaeology, the fairness audit,
the candidate proposal, the screen derivation, and the workload and compute
estimate. Then it stops.

Not authorised by this document: training any GNN, comparing anything against
QLS, opening M4, additional seeds, canonical CRAG, Package F, resuming E2, or
reopening QLS feature or semantic selection — which M2B closed.
