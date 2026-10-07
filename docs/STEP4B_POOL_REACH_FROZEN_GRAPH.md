# Step 4b: step 4's walk on each dataset's own expansion graph

Declared 7 October 2026, after step 4's verdict (docs/STEP4_POOL_REACH.md, Results) and before any number of this
step exists. It is a follow-up to step 4 of docs/MUSIQUE_DIAGNOSIS.md section 5, under that section's rule: one change,
made the same way on every dataset, graded on every dataset. Numbers here are development numbers; the paper's numbers
come from one declared confirmation run.

## 1. Question

Step 4 ran one walk, personalised PageRank from the frozen seeds, over the union of the three edge families on all six
graphs. At each question's frozen pool size it put all golds into the pool more often than the frozen pools on musique
(+0.032) and webqsp (+0.094), and less often on metaqa (−0.075), hotpotqa (−0.071) and 2wiki (−0.082) (step 4's
Results). The frozen expansions of musique and webqsp already run over the union of the three families. Those of the
other three use structural edges only, and one breadth-first rule over the union (step 4's U) was also below the frozen
pools there. So the union graph is the likely cause of the losses, not the walk:

- On hotpotqa and 2wiki, 0.91 and 0.88 of the golds that retrieval misses lie one structural hop from the seeds (M3A's
  reachability walk, `outputs/m3a/headroom/<dataset>.json`). A walk that spends two thirds of its steps on the other two
  families moves mass away from them.
- On metaqa, the golds of three-hop questions lie three structural hops out (0.49 within two, 1.0 within three). Step 4's
  walk lost there and gained on one- and two-hop questions.

Does the same walk, run on each dataset's own frozen expansion graph, put all of a question's golds into the pool more
often than the frozen breadth-first expansion, at the same pool size, question by question?

Against the frozen pools, only the expansion's order changes: PageRank mass instead of breadth-first order with
per-node caps and stored neighbour order, and no depth limit. The graph, the base list, the seeds and each question's
pool size stay the frozen ones. As in step 4, no model is trained or read.

## 2. The rule (P_F)

Step 4's P (docs/STEP4_POOL_REACH.md section 2) with one change: the walk's families are the frozen regime's, from
`m3b_compile.regime_families` (`configs/m3a_headroom.yaml`, `graph_regimes`):

| dataset | frozen regime | P_F walks |
| --- | --- | --- |
| metaqa | STRUCT | structural |
| squad | RETRIEVAL | nothing (no expansion) |
| musique | FULL | structural, ner, knn |
| hotpotqa | STRUCT | structural |
| 2wiki | STRUCT | structural |
| webqsp | FULL | structural, ner, knn |

- A family outside the regime enters step 4's kernel with no entries. The walk never picks it, and deg counts only the
  regime's entries. The kernel itself is step 4's, imported unchanged.
- Start, local push, ε = 1e-7, float64, dropped dangling mass, order (p descending, then node position, skipping base
  ∪ seeds) and P_F(B) are step 4's.
- **Restart α:** 0.25 or 0.5, chosen on the select carves by step 4's rule: the larger mean, over the five training
  datasets, of ΔALL(P_F − I) at matched size. A tie goes to 0.5. webqsp takes that value, so its read stays zero-shot.

On musique and webqsp, P_F is step 4's P. At α 0.5 their eval reads repeat step 4's, so they are not new evidence. This
file is written after seeing them, and section 6 says how they enter the verdict.

## 3. Comparator

- **I:** each dataset's frozen construction, built by `m3b_compile.prepare` unchanged.
- **Matched size, per question:** B_q = |I_q| − |base_q ∪ seeds_q|, as in step 4. A question whose walk offers fewer
  than B_q nodes is recorded as short.
- **Frontier:** B ∈ {0, 25, 50, 100, 200, 500, 1,000, 2,000} for P_F, beside the incumbent's point.
- Step 4's P over the union is reported beside P_F when step 4 filed it at the chosen α. It is not decided on.

## 4. Populations

Step 4's: step 1's `s1eval` carves for the reads, `s1sel` carves on the five training datasets for α only
(`outputs/step1/carves.json`, sha256 `53cfb89f…`, checked). No reserved carve and no test split is read.

## 5. Metrics

Step 4's: ALL, recall and ANY per question; per dataset and per slice (the step-1 stratum and the gold count); pool
size; entries read and wall time; paired P_F − I differences with a 95% paired bootstrap over questions (2,000
resamples, `numpy.random.default_rng(0)`).

## 6. Decision

Step 4's labels on ΔALL(P_F − I) at matched size and the chosen α: ABOVE if the interval's lower end is above 0, BELOW
if its upper end is below 0, AT otherwise. The verdict:

- **ADOPT:** no dataset BELOW and at least one ABOVE. A later declared step rebuilds the caches with P_F at the frozen
  sizes and retrains, graded on all six, zero-shot included. That step is the confirmation of this one.
- **NO_EFFECT:** all six AT.
- **NOT_ADOPTED:** any dataset BELOW. The pools stay as frozen.

squad has no expansion, so its label is AT by construction. If α is 0.5, musique and webqsp repeat step 4's reads and
labels. An ADOPT then rests on metaqa, hotpotqa and 2wiki not being BELOW, and the results say so.

## 7. What this does not do

- It trains nothing and reads no model or learned score.
- The encoder and the substrate embeddings are untouched.
- `m3b_pools`, `m3b_compile` and `scripts/step4_pool_reach.py` are imported and not edited.
- Step 4's files are not rewritten. The served package is read-only.
- It does not change any pool used by steps 1 to 3.

## 8. Run

`scripts/step4b_pool_reach.py` runs on the laptop (numba), with outputs in `outputs/step4b/`. It loads one dataset at a
time, and only the stores of that dataset's regime. Its three stages:

- `--stage select` covers the five select carves at both α and files α;
- `--stage eval` covers the six eval reads at the chosen α, with step 4's gold-free convergence check;
- `--stage grade` writes `outputs/step4b/result.json` and `result.md`.

Expected: about 10 minutes for select and 15 for eval. A single family has fewer entries than the union.

## Results

Filled in after the run.
