# Stage D0: does the restored graph information carry relevance signal?

Stage C established that `TARGET_H1 = Cq u N1_in(Cq)` performs its intended
repair. It did not establish that the repair is worth anything to a ranker.
D0 is the zero-training gate between those two claims: no model, no parameters,
no GPU, no test split. It computes four structural quantities under the
historical context `CAND = G[Cq]` and under `TARGET_H1`, on the same 300
validation queries with bit-identical candidate pools, and asks whether either
context separates relevant candidates from irrelevant ones better.

**The answer is not a clean yes.** Averaged over all candidates, the two
contexts are close, and the sign of the difference depends on which quantity you
look at: one of the four improves on 5 of 6 datasets by 0.010-0.066 AUC, one
*degrades* on 5 of 6, by as much as 0.274. The clean result is narrower and
more specific than "restoring the graph helps".

**What is clean.** On candidates that had no neighbour at all inside `G[Cq]`,
three of the four quantities are *identically zero* under `CAND` -- not weak,
not noisy, but constant, so their rank AUC is exactly 0.5000 on every query of
every dataset. Under `TARGET_H1` the same quantities become informative:
`bridge_support` reaches 0.9513 (2wiki), 0.9054 (musique) and 0.8749
(hotpotqa) mean AUC within that stratum. That is the repair doing the thing it
was built to do, converting *no signal at all* into signal.

**What bounds it.** That stratum is where gold candidates rarely live. A query
can only exercise it when at least one *relevant* candidate was isolated, and
that happens in 20.0% of 2wiki queries, 10.3% of musique, 6.0% of hotpotqa,
1.5% of webqsp, 1.0% of squad and 0.0% of metaqa. So the effect is real, is
largest exactly where Phase -1 said the damage was worst, and can only reach a
minority of queries.

| | |
|---|---|
| **Decision** | **ADVANCE TO THE SMALL D1 PILOT** -- 2wiki, one seed, `CAND` vs `TARGET_H1` |
| **Not** | a claim that the repaired context improves ranking. D0 has no ranker. |
| **Kill rule** | not met: the restored features do show relevance separation (§7) |
| **Prior for D1** | small positive or null. D0 predicts the addressable population on 2wiki is the 20.0% of queries with an isolated gold. |

---

## 1. What D0 is, exactly

| | |
|---|---|
| datasets | all six, validation split only |
| queries | 300 per dataset, the Stage-C sample, not a fresh draw |
| arms | `CAND` (historical substrate) and `TARGET_H1` |
| quantities | `seed_distance`, `distinct_seed_support`, `two_hop_seed_support`, `bridge_support` |
| statistic | per-query rank AUC, the pairwise win rate with ties at one half |
| labels | validation relevance labels, read **after** the pool and seeds, for scoring only |
| trained | nothing. No parameters, no GPU, no GNN. |
| evidence class | **development diagnostic, not an evaluation** |

Every result file carries the contract block asserting
`gold_ids_used_in_context_construction: false`, `test_split_read: false`,
`scored_nodes: exactly Cq in every arm`, and the frozen candidate-contract
hashes. `tests/test_graph_context_d0.py::test_context_construction_never_receives_a_label`
checks the label ordering at the source, because a diagnostic that leaks labels
into its own context would not look wrong in any output it produced.

A query with no positive or no negative candidate is **unscorable** and is
counted, never imputed at 0.5. That distinction matters here: 0.5 is also what a
constant scores, and "cannot discriminate" is not "is not defined".

## 2. Coverage, and the population the repair can reach

MEASURED:

| dataset | measured | one class only | median positives | queries with an isolated gold |
|---|---:|---:|---:|---:|
| 2wiki_clean | 300 | 0 | 2.0 | **60 (20.0%)** |
| musique_clean | 300 | 0 | 2.0 | **31 (10.3%)** |
| hotpotqa_clean | 300 | 0 | 2.0 | 18 (6.0%) |
| webqsp | 198 | 102 | 1.0 | 3 (1.5%) |
| squad_clean | 298 | 2 | 1.0 | 3 (1.0%) |
| metaqa | 195 | 105 | 1.0 | **0 (0.0%)** |

Two things have to be read off this table before any AUC is quoted.

**metaqa and webqsp lose a third of their queries.** 105 of 300 metaqa queries
and 102 of 300 webqsp queries have no negative candidate or no positive one, so
no pairwise comparison exists. Their AUCs are computed on 195 and 198 queries
and are not comparable in weight to the other four.

**The last column is derived, and it is the most decision-relevant number in
D0.** A degree stratum is scorable for a query exactly when it contains at
least one positive *and* one negative candidate. Isolated candidates are
17-37% of every pool (Phase -1), so a negative is essentially always present;
the stratum is therefore scorable exactly when a *gold* candidate was isolated.
On metaqa that never happened in 195 queries. On squad and webqsp it happened
three times each. Whatever the repair does for isolated candidates, it has
almost nothing to act on for ranking on those three datasets.

This is not a limitation of D0's design. It is a measurement of how much of the
retrieval problem the radius-1 truncation was actually breaking, and the answer
differs by an order of magnitude across datasets.

## 3. The two-path invariant, on real queries

Stage C proved that `TARGET_H1` preserves every directed `a -> v -> b` between
two candidates, with no symmetry assumption. D0 measured it on all six real
graphs. Kept/total, summed over 300 queries:

| dataset | CAND candidate 2-paths | TARGET_H1 | CAND seed 2-paths | TARGET_H1 | common successor lost |
|---|---:|---:|---:|---:|---:|
| 2wiki_clean | 36,090/157,440 (0.229) | 157,440/157,440 (**1.000**) | 4,099/8,765 (0.468) | 8,765/8,765 (**1.000**) | 0 |
| hotpotqa_clean | 42,599/206,254 (0.207) | 206,254/206,254 (**1.000**) | 8,168/22,436 (0.364) | 22,436/22,436 (**1.000**) | 0 |
| metaqa | 30,799/335,003 (0.092) | 335,003/335,003 (**1.000**) | 5,306/26,498 (0.200) | 26,498/26,498 (**1.000**) | 0 |
| musique_clean | 56,821/266,424 (0.213) | 266,424/266,424 (**1.000**) | 10,780/25,361 (0.425) | 25,361/25,361 (**1.000**) | 0 |
| squad_clean | 63,182/992,518 (0.064) | 992,518/992,518 (**1.000**) | 17,387/95,881 (0.181) | 95,881/95,881 (**1.000**) | 0 |
| webqsp | 38,105/96,777 (0.394) | 96,777/96,777 (**1.000**) | 7,897/12,850 (0.615) | 12,850/12,850 (**1.000**) | 0 |

The `1.000` column is the theorem and could not have come out otherwise; it is
reported because an identity that is never checked against the data is an
identity nobody has checked. The **`CAND` column is the finding**: vertex
induction on the candidate set destroyed **61% to 94%** of all two-hop paths
whose endpoints are both candidates, and **39% to 82%** of those starting at a
retrieval seed. That is a sharper statement of the historical damage than the
retention figures, because it is about paths -- the objects a two-layer message
passing network actually aggregates over -- rather than about edges.

**The exception the theorem allows does not occur here.** `TARGET_H1` can lose
an *undirected* two-path only in the orientation `a -> v <- b`, where the bridge
has no out-edge into `Cq`. Such bridges exist on hotpotqa -- its `TARGET_H1`
boundary cut is 0.0015, so some candidate out-edges do land outside the context
-- but no node in any of the six datasets is pointed at by two or more
candidates while pointing at none. The count is 0 on all six, MEASURED, not
assumed.

## 4. The negative control holds exactly

`distinct_seed_support` is identical under both contexts by construction: `Sq`
is inside `Cq`, so every seed-to-candidate edge already survives vertex
induction on `G[Cq]` and no wider context can add one.

MEASURED: the delta is **+0.0000 on all six datasets, and in every one of the
four degree strata**. The harness cannot manufacture a difference where none
exists. Every non-zero delta below is therefore a real difference in the
quantity, whatever its cause.

## 5. Discrimination over all candidates: small and mixed

Mean per-query rank AUC, `CAND` -> `TARGET_H1`:

| dataset | seed_distance | distinct_seed_support | two_hop_seed_support | bridge_support |
|---|---:|---:|---:|---:|
| 2wiki_clean | -0.0005 | +0.0000 | **-0.2742** | **+0.0664** |
| hotpotqa_clean | -0.0035 | +0.0000 | **-0.1821** | +0.0182 |
| metaqa | +0.0000 | +0.0000 | +0.0013 | +0.0103 |
| musique_clean | +0.0042 | +0.0000 | -0.0187 | +0.0231 |
| squad_clean | +0.0033 | +0.0000 | -0.0102 | **-0.0531** |
| webqsp | -0.0088 | +0.0000 | -0.0029 | +0.0186 |

`bridge_support` improves on five datasets and degrades on squad.
`two_hop_seed_support` degrades on five and improves on metaqa.
`seed_distance` does not move.

**These are point estimates with no error bar.** The runner aggregated the
per-query AUCs to a mean and a median and did not retain the paired per-query
differences, so D0 cannot say whether `+0.0664` on 2wiki is outside ordinary
validation noise. That is a real gap in the design and it is stated rather than
papered over: **no conclusion in this document rests on the sign of an overall
delta.** The result D0 does rest on (§7) is an exact identity on one side of the
comparison, not an estimate.

## 6. Why the widened context makes `two_hop_seed_support` *worse*

STRUCTURAL IDENTITY: under `CAND` the context is exactly `Cq`, so
`two_hop_seed_support` counts only the seeds that reach a candidate through a
bridge that is *itself a candidate*. Under `TARGET_H1` it counts seeds reaching
through any bridge in the corpus. The `CAND` version is not a noisy estimate of
the `TARGET_H1` version; it is a differently filtered quantity.

HYPOTHESIS, not tested here: the `CAND` filter is relevance-correlated.
"Reachable from a seed via another retrieved candidate" encodes something about
this query, because the pool was retrieved for this query. Widening to the true
graph replaces that with "reachable from a seed via anything", which is
dominated by high-degree corpus hubs and is largely query-independent. On 2wiki
the connected strata bear this out sharply -- `two_hop_seed_support` falls from
0.9510 to 0.5657 at degree 1 and from 0.9286 to 0.6399 at degree 2-4 -- while
the isolated stratum rises from 0.5000 to 0.5356.

The implication is not that the truncation was good. It is that **the
truncation was not simply a lossy version of the true graph: it was a biased
version, and its bias happened to point at relevance.** A count that is correct
about the corpus can be less useful than a count that is wrong about the corpus
but wrong in a query-specific direction. Any feature schema built on the
restored context has to earn back that accidental filtering explicitly --
normalisation, or an explicit candidate-mediated variant -- rather than assume
completeness is an improvement.

`bridge_support` is the one quantity that mostly survives the widening. It
counts distinct bridge *nodes* rather than distinct seeds, so it does not
saturate as fast when the graph is completed. squad is where it fails anyway:
squad has by far the densest contexts (992,518 candidate two-paths against
2wiki's 157,440) and its `bridge_support` drops from 0.9611 to 0.8423 at degree
2-4.

## 7. The isolated stratum: `0.5000` is an identity, not a measurement

This is the result.

A candidate isolated in `G[Cq]` has no neighbour inside the candidate set, and
`Sq` is inside `Cq`, so under `CAND` it has no seed edge and no bridge.
`distinct_seed_support`, `two_hop_seed_support` and `bridge_support` are
therefore **identically zero** on that stratum, and a constant scores rank AUC
exactly 0.5000 on every query. The `0.5000` cells in the `CAND` column are not
weak discrimination. They are the absence of a quantity to discriminate with.

VERIFIED FROM CODE against brute-force enumeration on 38 random graphs:
`tests/test_graph_context_d0.py::test_under_cand_every_support_is_identically_zero_on_isolated_candidates`.
MEASURED: the `CAND` isolated column reads exactly 0.5000 on all five datasets
that have a scorable isolated stratum, for all three quantities.

Mean per-query rank AUC within the isolated stratum, `CAND` -> `TARGET_H1`:

| dataset | queries | two_hop_seed_support | bridge_support |
|---|---:|---:|---:|
| 2wiki_clean | 60 | 0.5000 -> 0.5356 | **0.5000 -> 0.9513** |
| musique_clean | 31 | 0.5000 -> **0.8317** | **0.5000 -> 0.9054** |
| hotpotqa_clean | 18 | 0.5000 -> 0.6433 | 0.5000 -> 0.8749 |
| squad_clean | 3 | 0.5000 -> 0.6845 | 0.5000 -> 0.7080 |
| webqsp | 3 | 0.5000 -> 0.6670 | 0.5000 -> 0.6590 |
| metaqa | 0 | -- | -- |

**Only the first three rows carry weight.** squad and webqsp rest on three
queries each and are reported for completeness, not as evidence; metaqa has no
scorable isolated stratum at all. Quoting a five-dataset range for this effect
would be quoting two numbers that mean nothing.

On the three datasets where it can be measured, restoring the one-hop
neighbourhood turns a quantity that was constant-zero into one that ranks the
relevant isolated candidate above the irrelevant ones with 0.87-0.95 mean AUC.
Those are exactly the queries where the historical substrate could not have
ranked the gold candidate by message passing at all, because no message reached
it.

## 8. What `seed_distance` is measuring on that stratum, and why it is not this

`seed_distance` is the one column that is *not* pinned at 0.5000 under `CAND` on
isolated candidates -- it reads 0.9699 on 2wiki. That is not restored graph
signal and must not be read as any.

VERIFIED FROM CODE: an isolated candidate is unreachable and takes the
sentinel `max_hops + 1`, **unless it is itself a retrieval seed**, in which case
it is 0 by definition. So on the isolated stratum `seed_distance` ranks seeds
above non-seeds and nothing else; its AUC there is a retrieval-score proxy.
Pinned by
`tests/test_graph_context_d0.py::test_seed_distance_is_the_one_isolated_column_that_is_not_pinned_at_one_half`.

The same reading applies to `seed_distance`'s flat overall AUC of 0.86-0.98
across all six datasets at a tie fraction of 0.999-1.000: nearly every candidate
sits at the sentinel, the ordering is carried by the few that do not, and no
context wider than `SEED_H1` can move a frozen QLS-v1 distance bucket anyway.
That was measured in Stage C and is unchanged here.

The one exception worth recording: webqsp's isolated stratum moves 0.4938 ->
0.7244, the only place `seed_distance` moves materially. It rests on 3 queries.

## 9. Cost

MEASURED per-query wall time for the whole diagnostic -- context construction
*plus* the four quantities, which is what the timer brackets, so this is **not**
a context-construction cost:

| dataset | CAND p50 / p95 ms | TARGET_H1 p50 / p95 ms | marginal p50 |
|---|---:|---:|---:|
| musique_clean | 2.9 / 3.1 | 5.0 / 6.3 | +2.1 |
| metaqa | 5.9 / 6.0 | 11.6 / 13.6 | +5.7 |
| 2wiki_clean | 8.7 / 9.6 | 11.2 / 12.6 | +2.5 |
| squad_clean | 15.5 / 16.6 | 52.5 / 75.5 | +37.0 |
| hotpotqa_clean | 256.0 / 271.9 | 314.0 / 338.2 | +58.0 |
| webqsp | 259.0 / 325.5 | 307.1 / 391.5 | +48.1 |

The absolute figures on hotpotqa and webqsp are dominated by the diagnostic's
own sparse products over the whole node space, not by the context: `CAND` --
the *historical* substrate -- already costs 256-259 ms p50 under the same
measurement. The quantity that belongs to the graph-information regime is the
**marginal** column, +2.1 to +58.0 ms p50. This is a research diagnostic with no
optimisation effort behind it, and per the fairness contract amendment §9.1 it
says nothing about what the regime costs when computed properly from the global
CSR.

**Compute.** Six CPU containers, no GPU, no model. The launcher did not record
container elapsed time, so the spend is a **BOUND** from observed wall-clock
returns rather than a measurement: at most ~450 s per container, so <= 0.75
CPU-hours and <= $0.48 against the declared ceiling of <= 1.2 CPU-hours and
<= $0.80. `scripts/modal_graph_context_pilot.py` now returns `elapsed_seconds`
so that D1 reports a figure instead of a bound; that gap was the ledger's, not
the runner's, and is fixed.

## 10. The decision, against the rule as filed

The preregistered kill rule:

> If `TARGET_H1 <= CAND` within ordinary validation noise **and** the restored
> features show no useful relevance separation: DO NOT scale.

It is a conjunction, and the second clause is decisively false. On the three
datasets where the stratum can be measured, `bridge_support` under `TARGET_H1`
separates relevant from irrelevant isolated candidates at 0.87-0.95 mean AUC
where `CAND` has, by identity, no quantity at all. That is useful relevance
separation in restored information, and it is not a noise question, because one
side of the comparison is an exact constant.

The first clause is closer to true than not: over all candidates neither
context dominates. `bridge_support` moves at most 0.0664 in `TARGET_H1`'s
favour and 0.0531 against it; `two_hop_seed_support` moves up to 0.2742
*against* it; `seed_distance` does not move at all.
D0 therefore does **not** support "restoring the graph improves ranking". It
supports a much narrower statement:

> The information that candidate induction removed contains relevance signal
> that the historical substrate cannot express. Whether a ranker converts that
> into better retrieval is not answered here.

**Decision: ADVANCE TO THE SMALL D1 PILOT**, exactly as scoped -- 2wiki, one
seed, `CAND` vs `TARGET_H1`, same frozen ranker architecture and feature schema
in both arms, validation only. Nothing larger.

**The prior D1 is being run against, recorded before it runs.** 2wiki was
chosen before any D0 outcome, on Phase -1 grounds. D0 now says it is also the
dataset with the largest addressable population: 20.0% of its queries have an
isolated gold candidate, against 1.0% on squad and 0.0% on metaqa. The expected
effect is correspondingly bounded -- a small positive or a null. If D1 comes
back negative, D0's prediction is that the cause is §6: the connected strata
losing the truncation's accidental query-specific filtering by more than the
isolated stratum gains.

## 11. What D0 does not establish

* **Nothing about ranking.** No model was trained. There is no R@k, no MRR, and
  no answer yet to "does the repaired context improve retrieval". That is D1's
  question and D1 has not run.
* **Nothing about the four quantities as features.** They were chosen because
  they are computable without training, not because a ranker would use them.
  A schema that normalises by context size, or that keeps a candidate-mediated
  variant alongside the global one, might behave completely differently.
* **Nothing about significance.** No paired dispersion was recorded (§5).
* **Nothing about the backend.** Per fairness contract §9.1, the graph
  information privilege `TARGET_H1` and the QLS feature backend are separate
  objects, and the latency in §9 measures an unoptimised diagnostic.
* **Nothing about GNN depth.** Per §9.2 the contract is depth-aware and
  `TARGET_H1` is not asserted as the context for all depths.
* **Nothing about metaqa, squad or webqsp's isolated strata.** 0, 3 and 3
  scorable queries.

---

## 12. Full tables

Rendered by `scripts/render_d0_tables.py` from
`outputs/graph_context_pilot/stage_d0/`. Every number above is reproducible from
those files.
Split: `validation`. Datasets: `2wiki_clean`, `hotpotqa_clean`, `metaqa`, `musique_clean`, `squad_clean`, `webqsp`.

### Coverage

| dataset | measured | no seeds | one class only | median positives |
|:-------|-------:|-------:|-------------:|--------------:|
| 2wiki_clean | 300 | 0 | 0 | 2.0 |
| hotpotqa_clean | 300 | 0 | 0 | 2.0 |
| metaqa | 195 | 0 | 105 | 1.0 |
| musique_clean | 300 | 0 | 0 | 2.0 |
| squad_clean | 298 | 0 | 2 | 1.0 |
| webqsp | 198 | 0 | 102 | 1.0 |

### Two-path preservation, measured on real queries

The theorem says `TARGET_H1` keeps every directed `a -> v -> b` between candidates. Kept/total, summed over queries.

| dataset | CAND candidate 2-paths | TARGET_H1 | CAND seed 2-paths | TARGET_H1 | common successor lost |
|:-------|--------------------:|--------:|---------------:|--------:|-------------------:|
| 2wiki_clean | 36,090/157,440 (0.229) | 157,440/157,440 (1.000) | 4,099/8,765 (0.468) | 8,765/8,765 (1.000) | 0 |
| hotpotqa_clean | 42,599/206,254 (0.207) | 206,254/206,254 (1.000) | 8,168/22,436 (0.364) | 22,436/22,436 (1.000) | 0 |
| metaqa | 30,799/335,003 (0.092) | 335,003/335,003 (1.000) | 5,306/26,498 (0.200) | 26,498/26,498 (1.000) | 0 |
| musique_clean | 56,821/266,424 (0.213) | 266,424/266,424 (1.000) | 10,780/25,361 (0.425) | 25,361/25,361 (1.000) | 0 |
| squad_clean | 63,182/992,518 (0.064) | 992,518/992,518 (1.000) | 17,387/95,881 (0.181) | 95,881/95,881 (1.000) | 0 |
| webqsp | 38,105/96,777 (0.394) | 96,777/96,777 (1.000) | 7,897/12,850 (0.615) | 12,850/12,850 (1.000) | 0 |

### 2wiki_clean

**Discrimination over all candidates.** Mean per-query rank AUC.

| quantity | CAND | TARGET_H1 | delta | tie fraction | latency p95 ms |
|:-------|----:|--------:|----:|-----------:|------------:|
| seed_distance | 0.9790 | 0.9786 | -0.0005 | 0.999 | |
| distinct_seed_support | 0.7583 | 0.7583 | +0.0000 | 0.999 | |
| two_hop_seed_support | 0.8971 | 0.6228 | -0.2742 | 0.998 | |
| bridge_support | 0.9003 | 0.9667 | +0.0664 | 0.991 | |
| *context cost* | | | | | 9.6 -> 12.6 |

**Where the discrimination lands.** Mean per-query rank AUC within each stratum, by the induced degree the candidate had in `G[Cq]`.

| quantity | isolated CAND | TARGET_H1 | deg 1 CAND | TARGET_H1 | deg 2-4 CAND | TARGET_H1 | deg 5+ CAND | TARGET_H1 |
|:-------|----:|--------:|----:|--------:|----:|--------:|----:|--------:|
| seed_distance | 0.9699 | 0.9588 | 0.9681 | 0.9732 | 0.9725 | 0.9736 | 0.9393 | 0.9393 |
| distinct_seed_support | 0.5000 | 0.5000 | 0.6573 | 0.6573 | 0.7898 | 0.7898 | 0.8295 | 0.8295 |
| two_hop_seed_support | 0.5000 | 0.5356 | 0.9510 | 0.5657 | 0.9286 | 0.6399 | 0.8831 | 0.6158 |
| bridge_support | 0.5000 | 0.9513 | 0.9368 | 0.9542 | 0.9533 | 0.9612 | 0.9210 | 0.9191 |

### hotpotqa_clean

**Discrimination over all candidates.** Mean per-query rank AUC.

| quantity | CAND | TARGET_H1 | delta | tie fraction | latency p95 ms |
|:-------|----:|--------:|----:|-----------:|------------:|
| seed_distance | 0.9656 | 0.9622 | -0.0035 | 1.000 | |
| distinct_seed_support | 0.8559 | 0.8559 | +0.0000 | 0.998 | |
| two_hop_seed_support | 0.8933 | 0.7112 | -0.1821 | 0.999 | |
| bridge_support | 0.9224 | 0.9406 | +0.0182 | 0.985 | |
| *context cost* | | | | | 271.9 -> 338.2 |

**Where the discrimination lands.** Mean per-query rank AUC within each stratum, by the induced degree the candidate had in `G[Cq]`.

| quantity | isolated CAND | TARGET_H1 | deg 1 CAND | TARGET_H1 | deg 2-4 CAND | TARGET_H1 | deg 5+ CAND | TARGET_H1 |
|:-------|----:|--------:|----:|--------:|----:|--------:|----:|--------:|
| seed_distance | 0.8984 | 0.8850 | 0.9581 | 0.9443 | 0.9519 | 0.9480 | 0.9160 | 0.9175 |
| distinct_seed_support | 0.5000 | 0.5000 | 0.6922 | 0.6922 | 0.8162 | 0.8162 | 0.8405 | 0.8405 |
| two_hop_seed_support | 0.5000 | 0.6433 | 0.9041 | 0.6624 | 0.8610 | 0.6447 | 0.8096 | 0.6626 |
| bridge_support | 0.5000 | 0.8749 | 0.8533 | 0.9010 | 0.9202 | 0.9149 | 0.8892 | 0.8788 |

### metaqa

**Discrimination over all candidates.** Mean per-query rank AUC.

| quantity | CAND | TARGET_H1 | delta | tie fraction | latency p95 ms |
|:-------|----:|--------:|----:|-----------:|------------:|
| seed_distance | 0.9552 | 0.9552 | +0.0000 | 1.000 | |
| distinct_seed_support | 0.9781 | 0.9781 | +0.0000 | 0.997 | |
| two_hop_seed_support | 0.9831 | 0.9844 | +0.0013 | 0.998 | |
| bridge_support | 0.9642 | 0.9745 | +0.0103 | 0.986 | |
| *context cost* | | | | | 6.0 -> 13.6 |

**Where the discrimination lands.** Mean per-query rank AUC within each stratum, by the induced degree the candidate had in `G[Cq]`.

| quantity | isolated CAND | TARGET_H1 | deg 1 CAND | TARGET_H1 | deg 2-4 CAND | TARGET_H1 | deg 5+ CAND | TARGET_H1 |
|:-------|----:|--------:|----:|--------:|----:|--------:|----:|--------:|
| seed_distance | -- | -- | 0.9899 | 0.9899 | 0.9723 | 0.9723 | 0.7679 | 0.7679 |
| distinct_seed_support | -- | -- | 0.9820 | 0.9820 | 0.9473 | 0.9473 | 0.8848 | 0.8848 |
| two_hop_seed_support | -- | -- | 0.9933 | 0.9747 | 0.9742 | 0.9715 | 0.9112 | 0.9191 |
| bridge_support | -- | -- | 0.9688 | 0.9872 | 0.9327 | 0.9297 | 0.8257 | 0.8637 |

### musique_clean

**Discrimination over all candidates.** Mean per-query rank AUC.

| quantity | CAND | TARGET_H1 | delta | tie fraction | latency p95 ms |
|:-------|----:|--------:|----:|-----------:|------------:|
| seed_distance | 0.9514 | 0.9556 | +0.0042 | 1.000 | |
| distinct_seed_support | 0.7574 | 0.7574 | +0.0000 | 0.999 | |
| two_hop_seed_support | 0.8706 | 0.8519 | -0.0187 | 0.999 | |
| bridge_support | 0.8963 | 0.9194 | +0.0231 | 0.981 | |
| *context cost* | | | | | 3.1 -> 6.3 |

**Where the discrimination lands.** Mean per-query rank AUC within each stratum, by the induced degree the candidate had in `G[Cq]`.

| quantity | isolated CAND | TARGET_H1 | deg 1 CAND | TARGET_H1 | deg 2-4 CAND | TARGET_H1 | deg 5+ CAND | TARGET_H1 |
|:-------|----:|--------:|----:|--------:|----:|--------:|----:|--------:|
| seed_distance | 0.9162 | 0.9137 | 0.8811 | 0.8853 | 0.9531 | 0.9604 | 0.9208 | 0.9250 |
| distinct_seed_support | 0.5000 | 0.5000 | 0.5196 | 0.5196 | 0.6992 | 0.6992 | 0.7768 | 0.7768 |
| two_hop_seed_support | 0.5000 | 0.8317 | 0.8368 | 0.7551 | 0.8781 | 0.8182 | 0.7973 | 0.7780 |
| bridge_support | 0.5000 | 0.9054 | 0.8446 | 0.8625 | 0.9368 | 0.9239 | 0.8702 | 0.8656 |

### squad_clean

**Discrimination over all candidates.** Mean per-query rank AUC.

| quantity | CAND | TARGET_H1 | delta | tie fraction | latency p95 ms |
|:-------|----:|--------:|----:|-----------:|------------:|
| seed_distance | 0.9766 | 0.9799 | +0.0033 | 1.000 | |
| distinct_seed_support | 0.8591 | 0.8591 | +0.0000 | 0.998 | |
| two_hop_seed_support | 0.9026 | 0.8923 | -0.0102 | 0.999 | |
| bridge_support | 0.9268 | 0.8737 | -0.0531 | 0.957 | |
| *context cost* | | | | | 16.6 -> 75.5 |

**Where the discrimination lands.** Mean per-query rank AUC within each stratum, by the induced degree the candidate had in `G[Cq]`.

| quantity | isolated CAND | TARGET_H1 | deg 1 CAND | TARGET_H1 | deg 2-4 CAND | TARGET_H1 | deg 5+ CAND | TARGET_H1 |
|:-------|----:|--------:|----:|--------:|----:|--------:|----:|--------:|
| seed_distance | 0.8285 | 0.8154 | 0.9898 | 0.9898 | 0.9710 | 0.9788 | 0.9693 | 0.9701 |
| distinct_seed_support | 0.5000 | 0.5000 | 0.4796 | 0.4796 | 0.8678 | 0.8678 | 0.7515 | 0.7515 |
| two_hop_seed_support | 0.5000 | 0.6845 | 0.9286 | 0.8163 | 0.9436 | 0.9148 | 0.7960 | 0.7891 |
| bridge_support | 0.5000 | 0.7080 | 0.9592 | 0.9490 | 0.9611 | 0.8423 | 0.8745 | 0.8374 |

### webqsp

**Discrimination over all candidates.** Mean per-query rank AUC.

| quantity | CAND | TARGET_H1 | delta | tie fraction | latency p95 ms |
|:-------|----:|--------:|----:|-----------:|------------:|
| seed_distance | 0.8685 | 0.8597 | -0.0088 | 1.000 | |
| distinct_seed_support | 0.8053 | 0.8053 | +0.0000 | 0.996 | |
| two_hop_seed_support | 0.8120 | 0.8091 | -0.0029 | 0.998 | |
| bridge_support | 0.8384 | 0.8570 | +0.0186 | 0.989 | |
| *context cost* | | | | | 325.5 -> 391.5 |

**Where the discrimination lands.** Mean per-query rank AUC within each stratum, by the induced degree the candidate had in `G[Cq]`.

| quantity | isolated CAND | TARGET_H1 | deg 1 CAND | TARGET_H1 | deg 2-4 CAND | TARGET_H1 | deg 5+ CAND | TARGET_H1 |
|:-------|----:|--------:|----:|--------:|----:|--------:|----:|--------:|
| seed_distance | 0.4938 | 0.7244 | 0.7981 | 0.7965 | 0.8306 | 0.8267 | 0.7672 | 0.7634 |
| distinct_seed_support | 0.5000 | 0.5000 | 0.6847 | 0.6847 | 0.7540 | 0.7540 | 0.7595 | 0.7595 |
| two_hop_seed_support | 0.5000 | 0.6670 | 0.7900 | 0.7904 | 0.7422 | 0.7497 | 0.7260 | 0.7433 |
| bridge_support | 0.5000 | 0.6590 | 0.7208 | 0.8010 | 0.7670 | 0.7988 | 0.7590 | 0.7892 |

