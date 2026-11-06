# ABL: what each link family carries, with the models of record fixed, on every dataset

Declared 6 November 2026, before any ABL build, read or number. Redefined the same evening (this version replaces the
first declaration, commit 590f7a4, whose letters the user corrected before anything ran). Follows
[U2C_POOLS_ON_CHOSEN.md](U2C_POOLS_ON_CHOSEN.md), [B1P_PKO_CHAIN_ON_B1.md](B1P_PKO_CHAIN_ON_B1.md) (the b1t arm) and
[P2A_LINK_FAMILIES.md](P2A_LINK_FAMILIES.md). Code: `outputs/mp_unified/zabl.py` (committed after this file, before any
item runs). Outputs: only under `outputs/abl/`. No filed pool, fit, read or record changes.

## 1. Definitions (the user's ruling of 6 November, fixed here)

| letter | what it is | in the code |
| --- | --- | --- |
| **S** | the dataset's original graph links: the corpus hyperlinks (2Wiki, HotpotQA), the KB triples (MetaQA, WebQSP), whatever the package caches for MuSiQue and SQuAD. Never confused with N. | `structural` under graph "today" (the cached store) |
| **N** | our rule-built links (U2b's W2_S): the superseded NER evidence combined with the recreated hyperlinks; defined on the four passage datasets only (MetaQA and WebQSP have no passages to build it from) | the `structural` slot under graph "chosen" |
| **K** | nearest-neighbour links in the frozen encoder's space | `knn` |
| **O** | the old NER store that N superseded; still walked by the unified pool rule | `ner` |

What the filed record (SNK in the first declaration) actually holds, so nobody has to guess: on MuSiQue, SQuAD,
HotpotQA and 2Wiki the structural slot holds **N** (U2c replaced S by N), beside **O** and **K**; on MetaQA and WebQSP
the slot holds **S** (the KB), beside **O** and **K**. The record is therefore `N+O+K` on the passage datasets and
`S+O+K` on the KB datasets. The graph slot is written G below: G = N on passage datasets, G = S on KB datasets.

## 2. Arms

A removed family is an empty store (same node count, no edges) in every process that opens it. B, the seeds, the first
stage, the rule, its budgets and cuts are the record's. Tier 1 (declared, built here) is every subset of {G, O, K},
2^3 = 8, the record included; the models and walks need nothing new for any of them.

| arm | G | O | K | answers |
| --- | --- | --- | --- | --- |
| GOK | in | in | in | the record (ML1's reads, not rebuilt) |
| GK | in | out | in | remove the old NER |
| G | in | out | out | the user's "remove both NER and knn": the graph slot alone |
| OK | out | in | in | remove the slot: on passage datasets, what N adds (B1P's b1t on the development carves) |
| GO | in | in | out | remove knn |
| K | out | out | in | knn alone |
| O | out | in | out | the old NER store alone |
| 0 | out | out | out | the floor: first stage only |

Build order: GK, G, OK first (the user's asks), then GO, K, O, 0. On MetaQA and WebQSP "G out" means S out, which the user
allows for ablation only (S is never removed otherwise); those arms are built last.

**Tier 2, not declared here: S in the slot on the passage datasets** (S alone, S+K, S+O+K). PKO's pool walk is typed:
the EM and score types are fitted per graph (`zu2m_a_types.load_types`: "the types were fit on another graph"), and only
the types of N exist. S in the slot needs the types fitted on S first. That fit is its own stage (ABL-S), declared when
it is scoped. Beside it, B1H already filed S (the corpus hyperlinks, carve b1) against N (b1w) and the none arm (b1t)
for the three paper settings; those numbers exist and stay B1H's.

## 3. Carves, fits, reads

- **Carves:** the six development carves PKO builds (webqsp/s1eval, metaqa/s1eval, squad/s1eval, musique/s1eval,
  hotpotqa/s1eval, 2wiki/s1eval); 45,559 questions. Fit carves are not rebuilt; nothing is refitted.
- **Fits:** ML1's twelve pair-1 fits (zg6r, zrckp; seeds 0-2; WQ-musique, WQ-hotpotqa), forked without their reads to
  `outputs/abl/fits/<arm>/<fit>`. Each model reads the arm's pools, look, cache, links and KT in place of the record's.
  Both models read the same inputs.
- **Read:** ML1's read with `ML.ROOTS` and `ML.GATE` moved to `outputs/abl/<arm>/`; carves, regimes (in-domain, held
  out) and metric are the record's. R@5 on all questions.
- **Not in this stage:** the second pair (zrckpn, GNX) needs the next-hop columns rebuilt per arm: amendment A1, after
  IDP's GNX fits file. A refit on each arm's own fit carves is amendment A2 and the user's call. ABL's reads measure what
  the links carry for models fitted with them (B1P's "carry-over").

## 4. What is checked

1. B, the seeds and the first-stage lists of every question equal the record's (none reads a graph).
2. A removed family's store has zero edges wherever it is opened (asserted at load).
3. PKO's section 5 gate on the arm's carves as B1P ran it, the checks against U2s's arrays (graph-dependent by design)
   replaced by check 1; the replaced checks are named in the arm's gate record.
4. Rows per carve and arm are filed beside (costs only).
5. Before any arm's number is used, GOK re-read through ABL's own read path on webqsp/s1eval, seed 0, equals ML1's read
   to 1e-4.

## 5. Calls (declared before any number)

For each dataset, model, regime (in-domain, held out) and arm: R@5 difference in points, arm minus GOK, of the seed mean
(three seeds, both splits where the dataset is read under both), with a 95% interval from 2,000 paired bootstrap
resamples over questions (the seeds' per-question scores averaged). Call per cell: BELOW if the whole interval is below
zero, ABOVE if above, LEVEL otherwise; beside it, the number of the six fits where the arm is below GOK. Derived
columns, as differences of two arm cells with the interval from the paired per-question differences:

- what G adds = GOK - OK (all else in) and G - 0 (alone); what O adds = GOK - GK; what K adds = GOK - GO; what O and K
  add together beside the slot = GOK - G;
- the GNN - MLP difference per arm (does message passing matter more when fewer families are in).

The stage decides nothing about adoption; it files the grid and the calls.

## 6. Cost and order

PKOH measured 80 to 90 lane-hours for the whole 11-carve PKO build; the six development carves are 60% of its disk and
66% of its rows, so about 50 lane-hours for a record-sized arm, less for arms with fewer edges. Seven new arms are
about 250 to 350 lane-hours, 20 to 28 hours at 12 lanes, plus twelve GPU reads per arm beside the IDP fits. Per arm
the disk is up to 86 GiB (look 32 GiB of it, regenerated per arm because the rows differ): arms run one after another,
and an arm's look and cache are pruned once its twelve reads are filed (reads and records stay). Inside an arm the
carves go smallest first (webqsp, musique, hotpotqa, metaqa, squad, 2wiki), so a partial arm still yields cells.

## 7. Not claimed

- No refit: the numbers are the models of record on pools and edges with a family missing.
- The types arrays stay the record's. With the slot out the per-entry classes are empty (no entries), as in B1P's b1t.
- Development-split reads only; no test split is touched.
