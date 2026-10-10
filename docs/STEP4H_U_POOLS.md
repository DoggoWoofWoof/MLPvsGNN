# Step 4h: precise pools on the universal graphs

Declared 10 October 2026, about 01:35, before any number of the step on the universal graphs.

## Why

The user, 10 October: links first, then pools, then one retrain on both. Better hyperlink-like links, and pools with
full coverage "without having too many extra nodes".

Step 4g (docs/STEP4G_POOL_MISSES.md) found where step 4e's pools lose golds:
- **H1:** 78-89% of the missing golds on metaqa, musique and webqsp (30-42% on 2wiki) sit one hop from the pool, mostly
  over `structural` edges;
- **W:** 7-19% are reached by the walk but ranked below the cut;
- 51-92% are within two hops of a gold already in the pool.

On 2wiki and hotpotqa the reach runs over the corpus hyperlinks, which the headline must not use. Step 4h therefore
rebuilds the pools on U1d's universal graph (docs/U1D_PRECISE_LINKS.md) and adds one expansion aimed at H1.

## The pools

Each question's pool is its base and seeds (bs) plus m × B_q nodes ranked outside bs. B_q is step 4d's budget of the
question. The ranking is step 4e's chosen walk: arm A3, restart 0.5, the regime's families.

**Two graphs:**
- **GU:** `structural` is U1d's structural_U, built by U1d's chosen variant. These pools are the step's.
- **G0:** `structural` is today's family, the hyperlinks included on 2wiki and hotpotqa. They are reported beside GU
  as the reference and are never chosen.

ner and knn are unchanged on both. squad has no graph family: every arm is step 4e's RRF order.

**Two arms:**
- **W(m):** the walk's first m × B_q nodes.
- **B(m), m ≥ 2:** the walk's first (m − 1) × B_q nodes, then B_q bridge nodes. These are the nodes outside that pool
  adjacent, over the regime's families, to the question's top nodes: its seeds and the walk's first 10.
  - They are ranked by how many top nodes they touch, then by walk rank, then by position.
  - Fewer than B_q bridge nodes are topped up from the walk.
  - This targets H1 and the finding that missing golds sit next to found ones.

m ∈ {1, 2, 3, 4, 6}.

## The choice, fixed before any number

- **Reads:** the five s1sel carves and webqsp's s1eval carve (step 4e's six choice reads, amended 8 October).
- **Size cap:** a configuration (arm and m) is eligible only if its mean pool size, averaged over the six reads as a
  ratio to step 4e's chosen pools (A3, k = 2) on the same carve, is at most 1.0. No pool may be bigger, on average,
  than today's.
- **Criterion:** the highest mean ALL (every gold of the question in its pool) on GU among eligible configurations.
- **Tie:** within 0.002, the smaller ratio wins.
- **One choice for all six datasets.** No dataset gets its own m or arm.
- **Data read:** no question text beyond step 4e's linking (the plain question), and no model.

## What it reports

For every carve (s1sel and s1eval, all six datasets), on GU and G0:
- ALL, mean gold recall and mean pool size of every configuration;
- step 4e's filed numbers beside them.

The report states how much of each dataset's ALL GU keeps against G0. On 2wiki and hotpotqa that is the hyperlinks'
share of the pools' reach.

## What follows

U1c (its own file) retrains the GNN (zsp) and the MLP (zrc) on the chosen pools and the U graphs, and grades them on all
six datasets, zero-shot included, under the full-run null. B1 is read again beside it, never used to choose.

## How it runs

- `scripts/step4h_u_pools.py coverage --datasets D --host`, one host job per dataset (numba from the host's pylib), once
  U1d's build for D exists. Every job also runs G0, whose identity to steps 4c-4e checks the code on that dataset.
- Then `choose`, which writes outputs/step4h/choice.json and report.md.
- A laptop run on metaqa with G0 only precedes the host runs. It checks the code, not a number of the step.

## Amendment, 10 October about 01:50 (before any number on GU)

The laptop check on metaqa with today's graph (G0), which the file runs before the host, showed that the pools above
drop step 4e's frozen pool. Walk-only pools at 1.12 times step 4e's mean size held every gold on 0.910 of metaqa's s1sel
questions, against step 4e's 0.960. The frozen expansion holds golds the A3 walk ranks low. That is a defect of the
design, not a result of the step: no GU pool had been built.

**The pools now have step 4e's shape, every stage on the graph G (GU or G0):**
1. I^G: the frozen construction (m3b_compile.prepare, unchanged), with G's `structural` family.
2. U^G: I^G plus the first B_q nodes of step 4c's walk on G (step 4d's union pool).
3. W(k): U^G plus the A3 walk on G's first k x B_q nodes outside U^G, for k in {0, 0.5, 1, 2, 3} (step 4e's ks).
4. B(k), k in {1, 2, 3}: U^G plus the A3 walk's first (k - 1) x B_q nodes, then B_q bridge nodes. The bridge rule is
   the one above, with the top nodes being the seeds and the A3 walk's first 10.

B_q is step 4d's filed budget on both graphs.

**What changes, and what does not.**
- New check: on G0, I^G, U^G and every W(k) must equal steps 4c-4e (the frozen pools, step 4d's union pools, and step
  4e's filed coverage at every k). The stage stops otherwise.
- The choice is unchanged: the reads, the size cap (mean ratio at most 1.0 to step 4e's A3 k = 2), the highest mean
  ALL on GU, the 0.002 tie to the smaller pools, and one choice for all six.
- The configurations are now W0, W0.5, W1, W2, W3, B1, B2 and B3. W2 on GU is step 4e's chosen pool rebuilt on the
  universal graph.

## Results (filed 10 October 2026, about 07:25)

Run on the host (records: `outputs/step4h/{choice,coverage_<dataset>}.json`, `report.md`). On G0 every stage equals
steps 4c-4e on every dataset (the amendment's identity check).

**Choice reads** (the five s1sel carves and webqsp's s1eval, on GU):

| config | mean ALL | pool / step 4e | eligible |
| --- | ---: | ---: | --- |
| W0 | 0.8808 | 0.514 | yes |
| W0.5 | 0.9074 | 0.651 | yes |
| **W1** | **0.9172** | **0.787** | **yes, chosen** |
| B1 | 0.9120 | 0.787 | yes |
| W2 | 0.9263 | 1.060 | no (larger than step 4e) |
| B2 | 0.9293 | 1.060 | no |
| W3 | 0.9312 | 1.332 | no |
| B3 | 0.9360 | 1.332 | no |

**Chosen: W1 on GU**, at 0.79 of step 4e's mean pool size. U1c retrains on these pools.

**ALL / mean pool size, every carve:**

| dataset | carve | step 4e (G0, A3 k=2) | G0 W1 | **GU W1** | GU W2 |
| --- | --- | --- | --- | --- | --- |
| metaqa | s1sel | 0.960 / 7,093 | 0.956 / 5,122 | 0.947 / 5,099 | 0.949 / 7,071 |
| metaqa | s1eval | 0.953 / 7,082 | 0.949 / 5,115 | 0.938 / 5,091 | 0.941 / 7,058 |
| squad | s1sel | 0.992 / 150 | 0.989 / 100 | 0.989 / 100 | 0.992 / 150 |
| squad | s1eval | 0.992 / 150 | 0.989 / 100 | 0.989 / 100 | 0.992 / 150 |
| musique | s1sel | 0.950 / 6,921 | 0.934 / 5,028 | 0.937 / 4,966 | 0.954 / 6,859 |
| musique | s1eval | 0.937 / 6,899 | 0.921 / 5,006 | 0.924 / 4,937 | 0.940 / 6,829 |
| hotpotqa | s1sel | 0.978 / 188 | 0.975 / 144 | 0.905 / 167 | 0.908 / 211 |
| hotpotqa | s1eval | 0.979 / 188 | 0.976 / 144 | 0.896 / 166 | 0.905 / 210 |
| 2wiki | s1sel | 0.947 / 220 | 0.938 / 164 | 0.870 / 220 | 0.879 / 275 |
| 2wiki | s1eval | 0.945 / 221 | 0.936 / 166 | 0.873 / 222 | 0.882 / 278 |
| webqsp | s1eval | 0.909 / 7,213 | 0.893 / 5,293 | 0.857 / 5,249 | 0.877 / 7,169 |

### What it says

- **GU keeps musique's and squad's reach** (musique +0.003 over G0 at W1; squad equal). There the rule replaces
  today's structural family with no loss.
- **On 2wiki and hotpotqa GU loses 0.06-0.08 ALL against G0**, the hyperlinks' share of the pools' reach. The rule's
  links recover 0.57-0.68 of the hyperlinks at 0.23-0.32 precision (docs/U1D_PRECISE_LINKS.md); the pools pay for
  both the missed and the spurious links. This is the price U1c's rule names for those two datasets.
- **webqsp loses 0.036 ALL** (0.893 to 0.857 at W1). The mention links (mean degree 246 against the KB's 6.4) take the
  walk's mass away from the KB's triples. metaqa loses 0.011 for the same reason, on a much smaller scale.
- At the size cap no configuration on GU reaches step 4e's ALL on any dataset but musique (W2) and squad (W2).
- The next link fix (damping links to common-word titles and to over-linked targets, one rule for all six) is the
  universal answer to both losses. It needs its own file; U1c runs on today's choice.
