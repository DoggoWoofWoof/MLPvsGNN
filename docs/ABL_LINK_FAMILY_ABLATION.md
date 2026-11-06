# ABL: what each link family carries, with the models of record fixed, on every dataset

Declared 6 November 2026, before any ABL number. Redefined three times the same evening on the user's corrections
(first declaration 590f7a4; second bd1db44; third 2b4c9cd, amended 4a61bf1; this is the fourth). Follows
[U2C_POOLS_ON_CHOSEN.md](U2C_POOLS_ON_CHOSEN.md), [U2B_LINK_RULE.md](U2B_LINK_RULE.md),
[B1P_PKO_CHAIN_ON_B1.md](B1P_PKO_CHAIN_ON_B1.md) (the b1t arm) and [P2A_LINK_FAMILIES.md](P2A_LINK_FAMILIES.md). Code:
`outputs/mp_unified/zabl.py` (the arms, reads and report), `zabl_u.py` (the rule on the two KB datasets) and `zabl_g.py`
(the slot graphs and their types), each committed before it runs. Outputs: only under `outputs/abl/`. No filed pool,
fit, read or record changes.

## 1. Definitions (the user's rulings of 6 November, applied to all six datasets)

Three families, the same three on every dataset. A family is in or out as a whole.

| letter | what it is | stores it fills |
| --- | --- | --- |
| **S** | the original structural edges the dataset's graph store comes with, whatever they are | the `structural` slot, built from the store's own file (graph "today") |
| **R** | the rules as one family: the rule-built links (U2b's W2_S, the recreated hyperlinks) together with the NER store | the rule graph joins the `structural` slot; the NER store is the `ner` store |
| **K** | nearest-neighbour links in the frozen encoder's space | the `knn` store |

What each is, per dataset (edge counts are adjacency entries of the files the stage reads):

| dataset | S as shipped | R's rule graph | R's NER store | K |
| --- | --- | --- | --- | --- |
| 2Wiki | the corpus hyperlinks (5,989,847 nodes, 56.4M entries) | W2_S on the passages | the package's | the package's |
| HotpotQA | the corpus hyperlinks (5,233,329 nodes, 30.2M) | W2_S on the passages | the package's | the package's |
| MuSiQue | title-mention links (117,534 nodes, 5.3M) | W2_S on the passages | the package's | the package's |
| SQuAD | title-mention links (20,233 nodes, 1.49M; the store does ship one) | W2_S on the passages | the package's | the package's |
| MetaQA | the KB triples | **built here (section 8)**: the same rule on each entity's name and KB facts | the package's | the package's |
| WebQSP | the KB triples | **built here (section 8)** | the package's | the package's |

The record is not a cell on every dataset. The pools of record (U2c) run **R + K** on the four passage datasets (the rule
graph replaced S in the slot) and **S + NER + K** on the two KB datasets (no rule graph existed there). So SRK, with S and
the rule graph both in, is a new build on all six datasets. The passage record RK is ML1's reads and is not rebuilt; the
KB records are ML1's reads of a different pool and are kept beside the grid as a reference, not as a cell.

## 2. Arms

The eight subsets of {S, R, K}. A family that is out is an empty store (same node count, no edge) in every process that
opens it. B, the seeds, the first stage, the rule, its budgets and cuts are the record's. What fills the `structural`
slot: S alone is the original file; R alone is W2_S; S with R is the **union** of the two (undirected neighbours, rows
sorted by node id); neither is empty.

| arm | slot | ner | knn | note |
| --- | --- | --- | --- | --- |
| SRK | S ∪ R's graph | in | in | the full set; new build on all six |
| SR | S ∪ R's graph | in | out | K removed |
| SK | S | out | in | R removed |
| RK | R's graph | in | in | S removed; passage = the record (ML1's reads), KB = new build |
| S | S | out | out | |
| R | R's graph | in | out | |
| K | empty | out | in | |
| 0 | empty | out | out | the first stage only |

Build order (the user's asks first: take R out, take K out, take both out, from the full set): SRK, SK, SR, S, then RK
(KB), R, K, 0. The comparison base for every cell is SRK, on all six datasets.

## 3. Carves, fits, reads

- **Carves:** the six development carves PKO builds (`(ds, "s1eval")`: webqsp 1,503; musique 2,417; hotpotqa 7,405;
  metaqa 9,785; squad 11,873; 2wiki 12,576; 45,559 questions). Nothing is refitted.
- **Fits:** ML1's twelve pair-1 fits (zg6r, zrckp; seeds 0-2; WQ-musique and WQ-hotpotqa), forked without their reads to
  `outputs/abl/fits/<arm>/<fit>`. Each model reads the arm's pools, look, cache, links and KT in place of the record's.
  Both models read the same inputs.
- **Read:** ML1's read with `ML.ROOTS` and `ML.GATE` moved to `outputs/abl/<arm>/`; carves, regimes (in-domain, held
  out) and metric are the record's. R@5 on all questions. MetaQA has no in-domain fit (held out in both splits), WebQSP
  no held-out one, as in ML1.
- **Not in this stage:** the second pair (zrckpn, GNX) needs the next-hop columns rebuilt per arm (amendment A1, after
  IDP files). A refit on each arm's own pools is amendment A2 and the user's call. ABL measures what the families carry
  for models fitted on the record's pools (B1P's "carry-over").

## 4. What is checked

1. B, the seeds and the first-stage lists of every question equal the record's (none reads a graph).
2. A family that is out holds zero edges wherever it is loaded (asserted at load).
3. PKO's section 5 gate on the arm's carves as B1P ran it, the checks against U2s's arrays (graph-dependent by design)
   replaced by check 1; the replaced checks are named in the arm's gate record.
4. Rows per carve and arm are filed beside (costs only).
5. Before any number is used, the passage record re-read through ABL's own read path on musique/s1eval, seed 0, both
   models, equals ML1's read to 1e-4 (the GOK check; it moves from webqsp, whose record pool is no longer a cell).
6. The union slot graph: every row is the union of the two sources' rows (all rows checked), with its source files' sha256.
7. The KB rule graph (section 8): its build record names the node file's sha256, the text recipe's sha256 and the rule's
   code; beside it the share of its edges that are also KB triples (filed, not a gate).
8. The types of each slot graph are fit on that graph (never carried over from another), by the record's script
   unchanged (`zu2m_a_types.py`'s types stage, called by `zabl_g.py` with the slot graph in place of the record's): the
   EM centers on 200,000 pairs of the slot graph and every entry's class; the record's own files serve where the slot
   is the record's graph (R on the passage datasets, S on the KB). On the KB a slot that holds S's typed rows keeps
   them, and an entry only R's graph supplies has an empty block (the carried walk, RWe, weights an entry by its class,
   so no block is read). The passage score classes (`zu2p.py`) exist on the rule graph's pairs only: U2k scores a
   mention link or a sibling pair and refuses any other neighbour. So a slot's score class for a pair is the record's
   class where the rule graph holds the pair and the null class where it does not (S alone: every class null); the
   equal-count cuts over the scored entries are R's in every slot that holds R. Records name the graph file's sha256
   and the types files' sha256.

## 5. Calls (declared before any number)

For each dataset, model, regime (in-domain, held out) and arm: R@5 difference in points, arm minus SRK, of the seed mean
(three seeds, both splits where the dataset is read under both), with a 95% interval from 2,000 paired bootstrap resamples
over questions (the seeds' per-question scores averaged). Call per cell: BELOW if the whole interval is below zero, ABOVE
if above, LEVEL otherwise; beside it, the number of the six fits where the arm is below SRK. Derived columns, as
differences of two arm cells with the interval from the paired per-question differences:

- what S adds = SRK - RK; what R adds = SRK - SK; what K adds = SRK - SR; and each family's effect alone (S - 0, R - 0,
  K - 0);
- the GNN - MLP difference per arm (does message passing matter more when fewer families are in).

The stage decides nothing about adoption; it files the grid and the calls.

## 6. Cost and order

PKOH measured 80 to 90 lane-hours for the whole 11-carve PKO build; the six development carves are 60% of its disk and
66% of its rows, so about 50 lane-hours for a record-sized arm; arms with fewer edges cost less. Seven arms per
dataset (RK on passage is free): about 350 lane-hours in all, 30 hours at 12 lanes, plus the slot graphs and types (minutes
each, section 3 of the records) and the KB rule builds. Twelve GPU reads per arm beside the IDP fits. Disk is up to
86 GiB per full arm (look 32 GiB of it, regenerated per arm because the rows differ): arms run dataset by dataset
(chain, gate, twelve reads, prune of its look and cache), so a dataset's footprint is the peak. Inside an arm the carves
go smallest first (webqsp, musique, hotpotqa, metaqa, squad, 2wiki).

## 7. Not claimed

- No refit: the numbers are the models of record on pools and edges with a family added or missing.
- Development-split reads only; no test split is touched.
- On the KB datasets the rule can read only what the package holds there, the entity name and its KB facts, so its links
  are expected to be mostly KB neighbours already in S; the extras are the alias and sibling links and the NER store.
  That is what the rule can see there, not a finding about the datasets.

## 8. ABL-U: the rule on MetaQA and WebQSP

The user's ruling: R is the NER store and the recreated hyperlinks, applied exactly the same way to every dataset, on
whatever text exists. U2b's rule links a text to the nodes whose titles it names; a KB node's own text is its name, which
names nothing (U2b switch 4), so the rule has nothing to read until a node has text beyond its name. What exists in the
package for a KB entity is its name and its KB facts. So, as the declared recipe (changeable by the user, in which case
the KB arms are rebuilt):

- **Node text for the rule only** (the encoder's node vectors stay the name-only vectors of the substrate): the names of
  the entity's distinct KB neighbours, both directions, in node-id order, joined by "; " (its own name is left out, so no
  name is read as a mention of a shorter name inside it; an entity with no neighbour keeps its name alone and names
  nothing). The rule's boundary, forms, prefix and common-form switches, its AMBIG guard (50), its context choice by the frozen
  dense vectors and its variant W2 with the sibling links (the chosen graph, W2_S) are U2b's code, unchanged
  (`zu2b.Surfaces`, `stat_pass`, `link_pass`, `sibling_pairs`, `undirected_csr`).
- The capitalization statistic that gates common forms is computed over these texts, where every name is capitalized;
  the build record files how many forms it marks common (MetaQA: 149 of its surfaces).
- Output: `outputs/abl/u/<dataset>/W2_S.npz` with a record (node file sha256, the recipe's sha256, the code's sha256,
  edge counts, the share of edges that are KB triples).
- Then the slot graphs (S ∪ R's graph, R's graph) and their EM types (section 4, check 8). On the KB the slot graph keeps
  the KB's typed rows for S's entries; an entry only R's graph supplies has an empty typed run.
