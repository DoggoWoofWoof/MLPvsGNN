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

## Results

Not yet run.
