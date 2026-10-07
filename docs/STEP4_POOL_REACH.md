# Step 4: pool reach, one expansion rule on all six graphs

Declared 7 October 2026, before any number of this step exists. It is step 4 of docs/MUSIQUE_DIAGNOSIS.md section 5,
under that section's rule: a change made the same way on every dataset, graded on every dataset. Numbers here are
development numbers; the paper's numbers come from one declared confirmation run.

## 1. Question

Every question's pool is its dataset's frozen base list, its seeds, and a breadth-first expansion from the seeds. The
expansion's graph families, depth and caps were chosen per dataset (M3B's candidate contract,
`candidate_contract_frozen_2026_09_13`). It takes neighbours in stored order until a cap: structural neighbours by
node position, ner and knn neighbours by weight. Nothing in it looks at the question beyond the seeds.

| dataset | base | expansion | mean pool | all golds in the pool (eval) |
| --- | --- | --- | --- | --- |
| metaqa | rrf 50 | structural, 3 hops, 25 per node, 2,000 visited | 2,017 | 0.894 |
| squad | rrf 50 | none | 50 | 0.980 |
| musique | rrf 200 | all three families, 2 hops, 25 per node | 2,092 | 0.803 (four-hop questions 0.481) |
| hotpotqa | rrf 50 | structural, 1 hop, 25 per seed | 94 | 0.968 |
| 2wiki | rrf 50 | structural, 1 hop, 25 per seed | 106 | 0.906 |
| webqsp | dense 200 | all three families, 3 hops, 25 per node, 2,000 visited | 2,120 | 0.688 |

Almost every gold that retrieval's own pool misses lies within three steps of the seeds on the union of the three
families: 0.98 to 1.0 of them on every dataset, and 0.55 (metaqa) to 0.98 (2wiki) within two (M3A's reachability walk,
`outputs/m3a/headroom/<dataset>.json`, regime FULL). The golds are near; the expansion picks other nodes first.

Does one query-aware expansion rule, the same on all six graphs, put all of a question's golds into its pool more
often than the frozen expansion, at the same pool size, question by question?

The step measures pools only. No model is trained or read. A rule that passes goes to a later declared step that
rebuilds the caches on it and retrains, graded on all six.

## 2. The rule (P)

Personalised PageRank from the frozen seeds, as HippoRAG runs it from a question's entities.

- **Graph:** the union of the three families (structural, ner, knn), undirected, from the CSR stores M3B's pools read
  (`outputs/m3b/csr`, through `m3b_pools.load_or_build_store`). Edge weights and relation types are not used, so the
  rule runs on any graph.
- **Walk:** from node u, pick one of the families in which u has neighbours, uniformly, then one of u's neighbours in
  that family, uniformly.
- **Start:** uniform over the frozen seeds (dense top 5, then splade top 5, first occurrence kept).
- **Restart probability α:** 0.25 or 0.5, chosen on the select carves (section 4). HippoRAG uses 0.5.
- **Computation:** forward local push (Andersen, Chung and Lang, 2006). A node u is pushed while its residual is at least
  ε × max(deg(u), 1), where deg counts u's entries in the union. The queue is first in, first out, starting from the
  seeds in seed order. Arithmetic is float64. Mass reaching a node with no neighbour is dropped. The score is the
  pushed mass p. **ε = 1e-7.**
- **Order:** nodes outside base ∪ seeds with p > 0, by p descending, then node position ascending. P(B) = base ∪
  seeds ∪ the first B nodes of that order.

## 3. Comparators

- **I, the incumbent:** each dataset's frozen construction, built by `m3b_compile.prepare` unchanged.
- **U, one breadth-first rule:** `m3b_pools.expand_hops` over the union of the three families. It uses 3 hops, 25
  neighbours per family per seed and per frontier node, and a visited cap of |base ∪ seeds| + 2,010. Its order is
  expand_hops's own, with base ∪ seeds removed. U(B) = base ∪ seeds ∪ the first B. U answers whether a gain comes from
  P's scores or only from the union graph.
- **Matched size, per question:** B_q = |I_q| − |base_q ∪ seeds_q|. P and U are read at B_q, so each question's pool
  has its incumbent's size. A question whose rule offers fewer than B_q nodes is recorded as short.
- **Frontier:** B ∈ {0, 25, 50, 100, 200, 500, 1,000, 2,000} for P and U, beside each dataset's incumbent point.

The base list stays each dataset's own; only the expansion is one rule.

## 4. Populations

- **eval:** step 1's `s1eval` carves (`outputs/step1/carves.json`, sha256 `53cfb89f…`):
  - metaqa: 9,785 questions, every fourth of dev;
  - squad: 11,873 dev;
  - musique: 2,417 dev;
  - hotpotqa: 7,405 validation;
  - 2wiki: 12,576 dev;
  - webqsp: 1,503 train_holdout.

  Each carve's ids are checked against its recorded digest. The golds are `m3b_compile.population`'s eval golds.
- **select:** step 1's `s1sel` carves on the five training datasets, used only to choose α. α is the value with the
  larger mean, over the five datasets, of ΔALL(P − I) at matched size. A tie goes to 0.5. webqsp has no select carve
  and takes the α the other five chose, so its read is a zero-shot read of the rule.
- Neither stage touches a reserved carve (2wiki x2 and x3, musique x3) or a test split.

## 5. Metrics

- **Per question:**
  - ALL = 1 if every gold is in the pool;
  - recall = the share of its golds in the pool;
  - ANY = 1 if at least one gold is in the pool.
- **Per dataset and slice:**
  - the mean of each metric;
  - pool size: mean, p50, p95 and max;
  - expansion cost per question, as union entries read (deterministic) and wall time. Wall time is on the laptop, in
    numba, one thread per question, unpinned, so it is indicative only.
- **Slices:**
  - the step-1 stratum: metaqa and musique hop, squad answerable, hotpotqa level/type, 2wiki type;
  - gold count: 1, 2, 3, 4 or more.
- **Paired differences:** P − I and U − I per question. Each is reported as a mean with a 95% paired bootstrap interval
  over questions (2,000 resamples, `numpy.random.default_rng(0)`).

## 6. Decision

On the eval reads, at matched size and the chosen α, each dataset gets a label on ΔALL(P − I):

- **ABOVE** if the interval's lower end is above 0;
- **BELOW** if its upper end is below 0;
- **AT** otherwise.

The verdict:

- **ADOPT:** no dataset BELOW and at least one ABOVE. A later declared step rebuilds the caches with P at the incumbent
  sizes and retrains, graded on all six, zero-shot included.
- **NO_EFFECT:** all six AT.
- **NOT_ADOPTED:** any dataset BELOW. The pools stay as frozen.

squad has no expansion (B_q = 0), so its label is AT by construction. Its frontier shows whether its graph exposes any
gold.

U − I and the frontier are reported but not decided on. A bigger pool is a cost choice for a later step.

## 7. What this does not do

- It trains nothing and reads no model or learned score.
- The encoder and the substrate embeddings are untouched.
- `m3b_pools` and `m3b_compile` are imported and not edited.
- The served package is read-only.
- It does not change any pool used by steps 1 to 3.

## 8. Run

`scripts/step4_pool_reach.py` runs on the laptop: the walk needs numba, and the host has none. It loads one dataset at a
time and runs three stages:

- `--stage select` covers the five select carves at both α and files α;
- `--stage eval` covers the six eval reads at the chosen α;
- `--stage grade` writes `outputs/step4/result.json` and `result.md`.

Engineering measurements taken before this declaration used random seed nodes and read no gold. On 2wiki's union (5.99M
nodes, 152M entries):

- α 0.5, ε 1e-7: about 90 ms per question. The top 2,000 agree 0.93 with ε 1e-8.
- α 0.15, ε 1e-7: about 580 ms per question. The top 2,000 agree 0.86 with ε 1e-6.

The eval stage files the same gold-free check on real seeds: the top-2,000 overlap of ε with ε/10 on the first 20 eval
questions of each dataset.

Expected: select about 20 minutes, eval about an hour, grade minutes.

## Results

Run 7 October 2026, 06:31 to 06:52, on the laptop. Files: `outputs/step4/alpha.json`, `result.json` and `result.md`.

**Verdict: NOT_ADOPTED.** Restart value 0.5. On the select carves the pooled ΔALL(P − I) was −0.0455 at 0.5 and
−0.0483 at 0.25.

| dataset | frozen expansion graph | questions | ALL: I | P | U | ΔALL P − I [95%] | label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| metaqa | structural | 9,785 | 0.895 | 0.820 | 0.791 | −0.075 [−0.082, −0.068] | BELOW |
| squad | none | 11,873 | 0.980 | 0.980 | 0.980 | 0 | AT |
| musique | all three | 2,417 | 0.803 | 0.835 | 0.803 | +0.032 [+0.024, +0.042] | ABOVE |
| hotpotqa | structural | 7,405 | 0.968 | 0.897 | 0.946 | −0.071 [−0.078, −0.066] | BELOW |
| 2wiki | structural | 12,576 | 0.906 | 0.824 | 0.785 | −0.082 [−0.087, −0.077] | BELOW |
| webqsp (zero-shot) | all three | 1,503 | 0.688 | 0.782 | 0.688 | +0.094 [+0.074, +0.116] | ABOVE |

- **Where the frozen pools already expand over all three families, the walk wins:**
  - musique gains at every hop: two-hop +0.015, three-hop +0.054, four-hop +0.044 (0.481 to 0.526);
  - webqsp gains most on questions with four or more golds (0.290 to 0.501).

  On these two datasets U, breadth-first over the union, equals I exactly: their frozen expansions are already
  breadth-first over the union. The gain is the walk's order.
- **Where the frozen pools use structural edges only, both rules over the union lose.** So the union graph costs these
  datasets, not the walk:
  - hotpotqa: bridge questions 0.964 to 0.875;
  - 2wiki: bridge-comparison 0.769 to 0.518, compositional 0.926 to 0.866;
  - metaqa: one-hop +0.005 and two-hop +0.014, but three-hop 0.772 to 0.547.
- **Frontier: bigger walk pools reach more golds on every graph.** Examples: hotpotqa 0.976 at 250 nodes (I: 0.968 at
  94), 2wiki 0.932 at 250 (I: 0.906 at 106), musique 0.837 at 2,200, webqsp 0.786 at 2,200. A bigger pool is a cost
  choice, not decided here.
- **squad's graph exposes a few golds:** 0.980 at 50 nodes, 0.991 at 2,039.
- **Cost** (laptop, numba, one thread per question, unpinned):
  - the walk reads 0.30M to 0.75M union entries per question at the median, 13 to 70 thread-ms per question;
  - breadth-first takes 2.4 to 5.6 ms at the median;
  - convergence: the top-2,000 overlap of ε with ε/10 is 0.86 (musique) to 0.98 (metaqa).

These are development numbers. The pools stay as frozen. The next step, docs/STEP4B_POOL_REACH_FROZEN_GRAPH.md, runs the
same walk on each dataset's own expansion graph.
