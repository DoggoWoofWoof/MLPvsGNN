# Cross-Paper Learning Ledger

Status: **living record**. Last reconciled 2026-09-01.

Two lines of work share a corpus lineage and a set of hard-won lessons:

* **Paper 1** — this repository. *When Does Graph-Aware Retrieval Need Message
  Passing?* A post-retrieval candidate-ranking study over a frozen candidate
  contract. Its primary results are sealed.
* **Paper 2 / G2** — the C-RAG line, read-only from here. Retrieval and
  candidate construction itself: expansion, admission, partitioning, and
  end-to-end system behavior.

The ledger exists to stop a lesson learned on one side from silently leaking
into the other. **Paper 1 is frozen.** A finding that would change its
candidate pools, protocols, model inputs, or reported primary results is
recorded here and deliberately *not* backported. "Backport to Paper 1?" is
therefore almost always "No", and that is the intended answer, not an oversight.

`Reporting-only` means a finding may change how a Paper-1 number is *read* or
*presented* without changing anything that was computed.

## Ledger

| # | Finding | Origin | Backport to Paper 1? | Paper 1 role | Paper 2 / G2 role | Status |
|---|---|---|---|---|---|---|
| 1 | Fixed seed-distance and PPR summaries recover most of the benefit attributed to learned propagation | Paper 1, `P0_FIXED_STRUCTURAL_CONTROLS_*`, `NEURIPS_RESEARCH_PLAN.md` | Already in Paper 1 | Core Q3 control; establishes the QLS baseline that message passing must beat | Sets the floor any expansion mechanism must clear before learned propagation is justified | ESTABLISHED |
| 2 | A 19-parameter linear rank+structure model is a serious control, not a strawman | Paper 1, `P0_LINEAR_RANK_STRUCTURE_*` | Already in Paper 1 | Bounds how much of the effect is linear in rank and simple structure | Same control must gate any G2 candidate-scoring claim | ESTABLISHED |
| 3 | Structural vs embedding-kNN edge provenance are different mechanisms and must be separated | Paper 1 Package B, `EDGE_PROVENANCE_PROTOCOL.md`; C-RAG topology ablations | Already in Paper 1 (Package B) | Tests whether propagation and fixed structure respond differently to edge semantics | Chooses which graph a G2 expansion should walk | IN FLIGHT (B resumed 2026-09-01, 12/24) |
| 4 | Fixed features can leak dataset identity or labels unless audited | Paper 1, `SA_FEATURE_LEAKAGE_AUDIT.md`, `DATASET_GRAPH_PROVENANCE.md` | Already in Paper 1 | Label-free audit is a precondition for every QLS claim | Any G2 learned admission scorer needs the same audit before it is trusted | ESTABLISHED |
| 5 | Structural utility is regime-dependent, not a global property of a dataset | Paper 1 Packages A/E1 framing; C-RAG per-topology variation | Already in Paper 1 (E1 screens it) | Motivates the phase screen and the eventual utility predictor | Predicts where expansion pays for itself | IN FLIGHT (E1 resumed, 52/96) |
| 6 | Candidate generation, not reranking, is the binding constraint on several datasets | Paper 1, `CANDIDATE_HEADROOM_RESULTS.md` (2026-09-01) | **Reporting-only** | Every metric is reported beside `min(p,K)/g`; MetaQA attains 92% of its ceiling, WebQSP only 74% | The central quantitative motivation for G2: the headroom is upstream | ESTABLISHED |
| 7 | 92–100% of missing gold evidence sits within three hops of the frozen retrieval seeds | Paper 1, `CANDIDATE_HEADROOM_RESULTS.md` | **No — explicitly withheld** | Measured and reported; **no node admitted, no pool expanded** | Direct evidence that graph expansion is worth trying, and a quantified target | ESTABLISHED, DELIBERATELY NOT ACTED ON |
| 8 | The candidate ceiling rises with budget, so budget sweeps confound ceiling growth with reranking gain | Paper 1, headroom × Package C | **Reporting-only** | Package C effects reported as attainment against the per-budget ceiling | Any budget or admission sweep in G2 needs the same decomposition | ESTABLISHED |
| 9 | L1 partition/expansion behavior needs cross-dataset stability checks and true LODO, not single-split evidence | C-RAG `memory.md`, `LEVEL3_README.md` | No | Paper 1 uses fixed frozen splits and does not claim generalization across datasets | Governs the G2 expansion and partition-size study | ACTIVE IN PAPER 2 |
| 10 | Candidate admission is a distinct research question from candidate ranking | C-RAG `LEVEL3_README.md`, `README.md`; sharpened by Paper 1 finding #7 | **No — this is the Paper 1 / Paper 2 boundary** | Paper 1 ranks a frozen pool and says so explicitly | The G2 question proper | SCOPED TO PAPER 2 |
| 11 | Score normalization choice materially changes fusion and candidate ordering | C-RAG `memory.md`, `docs/level1_unified_protocol.md` | No | Paper 1 froze equal-RRF (constant 60, weights 0.5/0.5, ties by ascending global node ID) before any test outcome | Open design axis for G2 fusion | FROZEN IN PAPER 1, OPEN IN PAPER 2 |
| 12 | Directional / bidirectional bridge structure carries signal that undirected views discard | C-RAG `LEVEL3_README.md`, `docs/ablation_coverage_audit.md`; Paper 1 `COVERAGE_VARIANT_PROTOCOL.md` | No | Paper 1 reachability is measured on the **undirected** view and says so; the stored orientation is never modified | Candidate for a G2 typed/directed expansion operator | OPEN |
| 13 | Warm-cache and uncached costs are different results and must not be conflated | Paper 1, `RRF_AND_ONLINE_EVALUATION_FUTURE_WORK.md`, `ONLINE_SYSTEMS_PROTOCOL.md` | Already in Paper 1 | The 2.49–7.08x figure is labelled warm-cache only; Package D measures the uncached path | Any G2 system claim inherits the same separation | GATED (D waits on all six C budget-400 cells) |
| 14 | `modal run --detach` does not survive client teardown; long packages need server-side spawning | Paper 1 operations, 2026-09-01 | Already applied | `scripts/spawn_modal_jobs.py`; submission only, nothing scientific changed | Same failure mode applies to any long G2 run | RESOLVED |

## Backport discipline

1. A finding may enter Paper 1 only as **reporting**, never as a change to
   candidate pools, protocols, model inputs, hashes, or primary results.
2. Rows 7 and 10 are the load-bearing ones. Reachable missing golds are
   measured in Paper 1 and admitted only in Paper 2. Blurring that line would
   make Paper 1 a candidate-expansion paper it was never designed to be, and
   would invalidate its frozen contract.
3. No Paper-1 test outcome may be used to select anything in Paper 2 that is
   later compared back against Paper 1.
4. C-RAG stays strictly read-only from this repository.

## Mechanism ladder (future work, not Paper 1)

The natural progression once Paper 1 closes, each rung adding exactly one
capability so the attribution stays clean:

1. **Retrieval-only linear** — rank features alone, no structure.
2. **QLS linear** — fixed query-local structural summaries, linear head.
3. **QLS-MLP** — the same fixed summaries, nonlinear head. *(Paper 1 endpoint.)*
4. **Seed-aware GNN** — learned propagation over the candidate-induced graph.
   *(Paper 1 endpoint.)*
5. **Expansion-then-rank** — admit graph-reachable nodes before ranking.
   *(Paper 2 / G2. Row 7 is its motivation and its target.)*

Rungs 1–4 share a frozen candidate pool, which is what makes their differences
interpretable. Rung 5 breaks that invariant deliberately, which is exactly why
it belongs to a separate paper with its own frozen protocol.

## Gaps and provenance limits

Recorded rather than papered over:

* Rows 9, 11, and 12 are grounded in C-RAG documents located by topic, not by
  the exact phrasing used in the 2026-09-01 handoff directive. The underlying
  concepts are present; the precise result values were not re-verified from
  this repository, because C-RAG is read-only and its own protocol governs them.
* Rows 3 and 5 depend on Packages B and E1, which are resumed but incomplete.
  Their status must be re-read from `EXPERIMENT_EXECUTION_STATUS.md` before
  either row is cited as settled.
