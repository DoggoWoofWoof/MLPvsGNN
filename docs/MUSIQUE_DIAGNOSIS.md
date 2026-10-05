# MuSiQue diagnosis

**Declared 5 October 2026, before any number below was computed.** This file states the question, the inputs, the
slices and the metrics. The script is `scripts/musique_diagnosis.py`. Results are appended under "Results" after the
declaration is committed.

## Question

On musique the universal GAT reaches recall@5 0.521, against a pool ceiling@5 of 0.921. Where is that gap, and what
does the published number we calibrate against measure?

## Inputs

These are read-only, as filed by M3B (closed at 0bbb3c4). The script checks each sha256 against the value in
`docs/M3B_RESULTS.md` and refuses on a mismatch.

| file | sha256 |
| --- | --- |
| `outputs/m3b/eval/musique.npz` | `554b7a20605b6d965f3af62477103dc7c66e05079b716d781d2fa3d3408ee06c` |
| `outputs/m3b/eval/musique_query_ids.json` | `4412638ce9a31e412545fb750130f843f5abbdc8e52849cfe6dea840acbd1024` |

## Slices

- **Hop count**, from the MuSiQue question id prefix: `2hop` is 2, `3hop1` and `3hop2` are 3, and `4hop1`, `4hop2` and
  `4hop3` are 4.
- **Composition type**: the prefix itself.
- **All** 2,417 eval queries.

## Metrics

For each slice:

- n, the mean number of gold passages, the share of queries with every gold in the pool, and the pool ceiling@5. The
  ceiling is the mean of min(5, golds in the pool) / golds.
- recall@5, full_coverage@5 and hit@1 for each arm: fixed `rrf`, QLS-U, GAT-NO-MP and the universal GAT. Seed 0 is the
  row, and the mean over seeds 0 to 2 is shown beside it.
- δ_MP = GAT − GAT-NO-MP (seed 0) on recall@5 and full_coverage@5, with a paired bootstrap 95% interval over queries
  (2,000 resamples, seed 0).

The calibration is like for like. GraphER's PR@K is set coverage: 1 only if every gold passage is in the top K
(`docs/M3A_SOTA_ARCHAEOLOGY.md`). So our full_coverage@5 is placed next to GraphER's PR@5, not our recall@5.

## What this does not do

It trains nothing, selects nothing, and changes no filed record. It is a diagnosis, not a result of M3B, and M3B's
filed reading stands.

## Results (5 October 2026, after the declaration at 1caa2fa)

The diagnosis output is `outputs/musique_diag/diagnosis.json`. The carve audit (section 4) is
`scripts/musique_carve_audit.py`, which writes `outputs/musique_diag/carve_audit.json`; it only counts question ids.

### 1. What the published numbers measure

| system | metric | MuSiQue | setup |
| --- | --- | --- | --- |
| GraphER: MLP / GCS / GAT | PR@5: every gold in the top 5 | 21.6 / 25.4 / 25.6 | 2,000 sampled dev questions, a corpus induced from them, 200 candidates, trained per dataset |
| ours: `rrf` / QLS-U / GAT-NO-MP / GAT | full_coverage@5, the same metric | 13.5 / 18.0 / 17.5 / 22.4 | all 2,417 dev questions, the full 117,534-passage corpus, 2,092 candidates, one model for six datasets |
| HippoRAG 2 / NV-Embed-v2 alone / HippoRAG / Contriever / BM25 | recall@5 | 74.7 / 69.7 / 53.2 / 46.6 / 43.5 | 1,000 dev questions over 11,656 passages. NV-Embed-v2 (7B) is the retriever and Llama-3.3-70B builds the graph (HippoRAG 2, arXiv 2502.14802, Table 3). |
| ours: `rrf` / QLS-U / GAT-NO-MP / GAT | recall@5 | 47.3 / 49.1 / 50.3 / 52.1 | as above |

GCS is GraphER's parameter-free arm, a linear smoothing of the retriever scores.

**Correction to M3B section 6.** That calibration row put our recall@5 (0.521) beside GraphER's PR@5. PR@K is set
coverage, so the like-for-like number is our full_coverage@5, 0.224.

- That is still at or above the lowest published number (0.216), so the band verdict READ stands. It stands narrowly,
  and it sits below GraphER's GAT (0.256) and its parameter-free arm (0.254).
- The same correction leaves hotpotqa (0.838 against 0.780 to 0.789) and 2wiki (0.756 against 0.425 to 0.441)
  unchanged.
- M3B's sealed record is not edited. This note is the correction.

On recall@5, ours (0.521) sits at the original HippoRAG's level and 22.6 points below HippoRAG 2. Most of HippoRAG 2's
number comes from its 7B embedding model, which alone reaches 69.7. Its corpus is also a tenth the size of ours.

### 2. Where the gap is: questions with three and four hops

Seed 0. δ_MP = GAT − GAT-NO-MP, with the paired bootstrap 95% interval.

| hops | questions | golds | all golds in the pool | ceiling@5 | R@5: `rrf` / QLS-U / GAT-NO-MP / GAT | FC@5: GAT | hit@1: GAT | δ_MP on R@5 | δ_MP on FC@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1,252 (52%) | 2 | 0.941 | 0.966 | 0.562 / 0.593 / 0.597 / **0.643** | 0.411 | 0.679 | **+0.046** [+0.032, +0.060] | **+0.101** [+0.077, +0.125] |
| 3 | 760 (31%) | 3 | 0.746 | 0.909 | 0.437 / 0.445 / 0.462 / 0.445 | 0.034 | 0.642 | −0.018 [−0.032, −0.004] | −0.009 [−0.026, +0.008] |
| 4 | 405 (17%) | 4 | 0.481 | 0.802 | 0.269 / 0.262 / 0.290 / 0.286 | 0.000 | 0.578 | −0.004 [−0.017, +0.010] | +0.000 |
| all | 2,417 | 2.65 | 0.803 | 0.921 | 0.473 / 0.491 / 0.503 / 0.521 | 0.224 | 0.650 | +0.018 [+0.008, +0.027] | +0.049 [+0.036, +0.062] |

- **Message passing helps only the two-hop questions.** The whole of musique's δ_MP comes from them. On three-hop
  questions it is slightly below zero, and on four-hop questions it is zero. The GAT has two message-passing layers.
- **The first passage is usually found; the later ones are not.** hit@1 stays at 0.58 to 0.68 at every hop count, but
  almost no three-hop question and no four-hop question gets all its passages into the top 5.
- **The pool misses the far hops.** It expands two graph steps from the seeds. A four-hop question has every gold in
  the pool only 48% of the time.

### 3. Training rewards a shortcut that dev does not have

MuSiQue composes its training questions from shared single-hop questions. Its dev split uses none of them: 0 of dev's
2,768 single-hop components appear in train.

| population | holds a fit gold | holds only fit golds | gold passages that are fit golds | shares a single-hop component with a fit question |
| --- | --- | --- | --- | --- |
| select carve (1,534) | 0.945 | 0.411 | 0.622 | 0.941 |
| dev (2,417) | 0 | 0 | 0 | 0 |

So the select carve, which chose each arm's epoch and configuration, rewarded remembering passages that were gold in
training. Dev never rewards that.

- **On the select carve**, the learned arms reach R@5 0.745 to 0.761, against 0.538 for `rrf`.
- **On dev** they reach 0.491 to 0.521, against 0.473.
- **The part that does not carry** is 0.18 to 0.19 of R@5, beyond `rrf`'s own drop of 0.064 (M3B section 4b).

The hop mix shifts too:

| split | 2-hop | 3-hop | 4-hop |
| --- | --- | --- | --- |
| train | 72% | 22% | 6% |
| dev | 52% | 31% | 17% |

### 4. A component-disjoint carve is feasible

Training questions link to each other through shared single-hop components. Following those links, the 19,938
training questions fall into 1,012 groups, and the largest holds 2,231 questions (11%).

So whole groups can be assigned to the select carve. Then no select question shares a component with a fit question,
which is the relation dev has to train. No question has to be dropped. Holding out random components instead would drop
10% to 48% of the training questions.

### 5. What would improve the training, for every dataset

**Rule (5 October 2026).** Every step below is a change to the training of all six datasets, graded on all six. None
is a musique-only fix. musique is where the causes showed up, but a step counts only if every dataset gains,
generalization included. These steps are proposed, not run, and each needs its own declared file before any number.

1. **Distribution-matched selection on every dataset.** Each dataset's select carve gets the same overlap with the fit
   carve as its eval population has, and the same hop or difficulty mix. On musique that means whole linked groups,
   since dev shares no component with train. On squad it means no shared gold. On the KBs it keeps their natural
   entity overlap. Selection then rewards what each eval rewards. This is the cheapest step, and it comes first.
2. **A difficulty-balanced fit on every dataset.** Weight long chains up to each eval's mix: musique's three- and
   four-hop questions, metaqa's three-hop questions, and multi-gold questions on hotpotqa and 2wiki.
3. **Depth in the universal models.** Use three or four message-passing layers, or read-time rounds that go back to
   the neighbours of the top candidates. That is S6's idea of going back, and HippoRAG's personalised PageRank does
   the same from the question's entities. Grade it on every dataset's multi-hop slices and on the zero-shot reads.
4. **Pool reach on every graph.** Apply one expansion rule to all six graphs, and measure the all-golds-in-the-pool
   share against the cost. musique's four-hop share is now 0.48.

**The encoder is not on this list and never will be.** It stays frozen with the substrate; Swastik ruled this out on 5
October 2026. The part of the gap to HippoRAG 2 that comes from its 7B encoder is a named difference in exposure, and
nothing here tries to close it.
