# ABL: what each link family carries, with the models of record fixed, on every dataset

Declared 6 November 2026, before any ABL build, read or number. Redefined twice the same evening on the user's
corrections (first declaration 590f7a4; second bd1db44; this is the third). Follows
[U2C_POOLS_ON_CHOSEN.md](U2C_POOLS_ON_CHOSEN.md), [B1P_PKO_CHAIN_ON_B1.md](B1P_PKO_CHAIN_ON_B1.md) (the b1t arm) and
[P2A_LINK_FAMILIES.md](P2A_LINK_FAMILIES.md). Code: `outputs/mp_unified/zabl.py` (committed after this file, before any
item runs). Outputs: only under `outputs/abl/`. No filed pool, fit, read or record changes.

## 1. Definitions (the user's rulings of 6 November, fixed here)

| letter | what it is | in the code |
| --- | --- | --- |
| **S** | the dataset's original graph links: the corpus hyperlinks (2Wiki, HotpotQA), the KB triples (MetaQA, WebQSP), whatever the package caches for MuSiQue and SQuAD | `structural` under graph "today" |
| **R** | the rules, one combined family: our rule-built links (U2b's W2_S: a text links to a passage whose title it names under the rule's forms; the "recreated hyperlinks") together with the NER store they superseded | the `structural` slot under graph "chosen" **plus** the `ner` store |
| **K** | nearest-neighbour links in the frozen encoder's space | `knn` |

Where R is built today (one fact the table hides): U2b's rule graph exists on MuSiQue, SQuAD, HotpotQA and 2Wiki only. On
MetaQA and WebQSP the graph is the KB itself, there is no rule-built graph, and the `ner` store is all of R there. So:

- **Passage datasets (4):** the record is `R + K`; S is not in it (U2c replaced S by the rule graph).
- **KB datasets (2):** the record is `S + R + K` with R = the NER store alone; S is the KB.

The same arms and letters apply to all six datasets; what differs is what R holds, and it is stated in every table.
Applying the rule itself to the KB datasets (building rule links on their node text) changes their pools and is a new
stage, **ABL-U**, declared after the arms below are queued (section 8).

## 2. Arms

A family that is out is an empty store (same node count, no edge) in every process that opens it. B, the seeds, the first
stage, the rule, its budgets and cuts are the record's. R is always in or out as a whole (slot and NER store together
on passage datasets). Arm names in `zabl.py` are the three stores G (slot), O (ner), K (knn); the table maps them.
The arms K and 0 are built on the passage datasets under those names and on the KB datasets (S-out, ablation only) as
`Kb` and `0b`, so that each arm's gate, reads and folders cover one set of datasets.

| arm (zabl) | passage datasets (R = rule graph + ner) | KB datasets (S slot, R = ner) |
| --- | --- | --- |
| GOK | record: R + K (ML1's reads, not rebuilt) | record: S + R + K |
| GO | R only (K out) | S + R (K out) |
| K | K only (R out) | K only (S and R out; ablation only) |
| 0 | first stage only | first stage only (ablation only) |
| GK | not built (G and O move together) | S + K (R out) |
| G | not built | S alone (R and K out) |
| OK | not built | R + K (S out; ablation only) |
| O | not built | R alone (S and K out; ablation only) |

Build order: GO on all six, GK and G on the KB datasets, K and 0 on the passage datasets (the user's asks first: remove
R, remove K, remove both), then the S-out arms on the KB datasets (K, 0, OK, O), which exist only because the user allows
removing S for ablation. **Tier 2, not declared here: S in the slot on the passage datasets** (S alone, S + K, S + R):
PKO's pool walk is typed, the EM and score types are fitted per graph (`zu2m_a_types.load_types`: "the types were fit on
another graph") and exist for the rule graph only, so S in the slot needs its types fitted first (a stage of its own).
B1H already filed S (the corpus hyperlinks, carve b1) against the rule graph (b1w) and none (b1t) on the three paper
settings; those numbers stay B1H's.

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
  the families carry for models fitted with them (B1P's "carry-over").

## 4. What is checked

1. B, the seeds and the first-stage lists of every question equal the record's (none reads a graph).
2. A family that is out holds zero edges wherever it is loaded (asserted at load).
3. PKO's section 5 gate on the arm's carves as B1P ran it, the checks against U2s's arrays (graph-dependent by design)
   replaced by check 1; the replaced checks are named in the arm's gate record.
4. Rows per carve and arm are filed beside (costs only).
5. Before any arm's number is used, GOK re-read through ABL's own read path on webqsp/s1eval, seed 0, equals ML1's read
   to 1e-4 (`zabl.py items --arm GOK`: fork, two reads, `check`, for zg6r and zrckp; the check files its result).

## 5. Calls (declared before any number)

For each dataset, model, regime (in-domain, held out) and arm: R@5 difference in points, arm minus GOK, of the seed mean
(three seeds, both splits where the dataset is read under both), with a 95% interval from 2,000 paired bootstrap
resamples over questions (the seeds' per-question scores averaged). Call per cell: BELOW if the whole interval is below
zero, ABOVE if above, LEVEL otherwise; beside it, the number of the six fits where the arm is below GOK. Derived columns,
as differences of two arm cells with the interval from the paired per-question differences:

- what R adds = GOK - K (passage) and GOK - GK / G - 0 on the KB datasets; what K adds = GOK - GO; what R and K add
  together beside S (KB datasets) = GOK - G;
- the GNN - MLP difference per arm (does message passing matter more when fewer families are in).

The stage decides nothing about adoption; it files the grid and the calls.

## 6. Cost and order

PKOH measured 80 to 90 lane-hours for the whole 11-carve PKO build; the six development carves are 60% of its disk and
66% of its rows, so about 50 lane-hours for a record-sized arm, less for arms with fewer edges. The passage datasets are
44% of that (about 22 lane-hours per arm: three arms), the KB datasets 56% (about 28 per arm: three first, four ablation
only). First tier (GO on six, GK and G on the KB, K and 0 on the passage): about 150 lane-hours, 12 to 13 hours at 12 lanes;
the S-out arms add about 110. Twelve GPU reads per arm beside the IDP fits. Disk is up to 86 GiB per full arm (look 32 GiB
of it, regenerated per arm because the rows differ): arms run one after another and an arm's look and cache are pruned
once its twelve reads are filed (reads and records stay). Inside an arm the carves go smallest first (webqsp, musique,
hotpotqa, metaqa, squad, 2wiki).

## 7. Not claimed

- No refit: the numbers are the models of record on pools and edges with a family missing.
- The types arrays stay the record's. With the slot out the per-entry classes are empty (no entries), as in B1P's b1t.
- Development-split reads only; no test split is touched.

## 8. ABL-U: the rules on every dataset (declared as a follow-up, not run here)

The user's ruling: the same rules should apply universally. Today the rule graph is built on four datasets. ABL-U would
build U2b's rule on the node text of MetaQA and WebQSP (the KB entities: name and facts), file it as R's slot there, and
rebuild their pools and look, so that R is the same object on all six. It changes those datasets' pools and therefore
what the models read: it is a development stage with its own adoption call, scoped when ABL's first tier is queued,
and not a read of the record.
