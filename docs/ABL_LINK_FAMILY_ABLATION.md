# ABL: what each link family carries, with the models of record fixed, on every dataset

Declared 6 November 2026, before any ABL build, read or number. Follows [U2C_POOLS_ON_CHOSEN.md](U2C_POOLS_ON_CHOSEN.md)
(the three families, the chosen graph), [B1P_PKO_CHAIN_ON_B1.md](B1P_PKO_CHAIN_ON_B1.md) (the b1t arm: the structural
family removed, ner and knn kept, the models fixed) and [P2A_LINK_FAMILIES.md](P2A_LINK_FAMILIES.md) (the families at
the level of reach). Code: `outputs/mp_unified/zabl.py` (committed after this file, before any item runs). Outputs:
only under `outputs/abl/`. No filed pool, fit, read or record changes.

## 1. The question

The user's ruling of 6 November, in their words: the links come in three families, "structural over the whole graph or
the original links, NER which we superseded and combined with the recreated hyperlinks to create the rule based links,
and knn"; "we just want to say remove NER, how much do the links gain us, might as well remove both the NER and knn and
get the numbers too, so we can do a proper ablation with the same inputs over 6 datasets, of both the models, both in
domain and generalization type".

In the repository the three families are `structural` (S: the KB triples on MetaQA and WebQSP; on the four passage
datasets U2b's chosen rule W2_S, which sits in the structural slot and is the rule-built link family), `ner` (N) and
`knn` (K). The unified pool rule walks all three on every dataset (U2s, U2m). B1P already filed one arm on three
settings (b1t: S out, N and K kept); ABL files every arm on all six development carves.

## 2. Arms

A removed family is replaced, in every process that opens it, by an empty store with the same node count (no edges).
Nothing else is changed: the first stage, the base pool B, the seeds, the rule, its budgets and its cuts are the
record's.

| arm | S | N | K | what it answers |
| --- | --- | --- | --- | --- |
| SNK | in | in | in | the record (no build; ML1's reads) |
| NK | out | in | in | what the rule-built family adds (B1P's b1t, on the development carves) |
| SK | in | out | in | what NER adds beside the other two |
| S | in | out | out | the user's "remove both NER and knn": the rule links alone |
| SN | in | in | out | what knn adds |
| 0 | out | out | out | the floor: first stage, no graph at all |

Build order: SK, S, NK (the user's three), then SN, 0. Single-family arms N and K are not declared.

## 3. Carves, fits, reads

- **Carves:** the six development carves PKO builds (webqsp/s1eval, metaqa/s1eval, squad/s1eval, musique/s1eval,
  hotpotqa/s1eval, 2wiki/s1eval); 45,559 questions. Fit carves are not rebuilt: nothing is refitted.
- **Fits:** ML1's twelve pair-1 fits (zg6r and zrckp, seeds 0-2, WQ-musique and WQ-hotpotqa), copied without their
  reads to `outputs/abl/fits/<arm>/<fit>` (B1P's fork). Each model reads the arm's pools, look, cache, links and KT in
  place of the record's. Both models read the same inputs (ML1's identical-input pair).
- **Read:** ML1's read with `ML.ROOTS` and `ML.GATE` moved to `outputs/abl/<arm>/`; the carves, the regimes (in-domain,
  held out) and the metric are the record's. R@5 on all questions of each carve.
- **Not in this stage:** the second pair (zrckpn, GNX) needs the next-hop columns rebuilt per arm; that is amendment A1,
  declared after IDP's GNX fits file. A refit on each arm's own fit carves (the models then see the same family set at
  training and at reading) is amendment A2 and the user's call; ABL's reads measure what the links carry for models
  fitted with them (B1P's "carry-over").

## 4. What is checked

1. The base pool B, the seeds and the first-stage lists of every question equal the record's (they do not read a graph).
2. The removed family's store has zero edges wherever it is opened (asserted at load).
3. PKO's section 5 gate on the arm's six carves, as B1P ran it, with the checks that compare a pool to U2s's arrays
   (graph-dependent by design) replaced by check 1; the replaced checks are named in the arm's gate record.
4. Row counts per carve and arm are filed beside (costs only).
5. The arm SNK's cells are ML1's filed reads, not re-read; one arm-free re-read (SNK through ABL's own read path on
   webqsp/s1eval, seed 0) must equal ML1's read to 1e-4 before any arm's number is used.

## 5. Calls (the rule is declared here; the numbers come later)

For each dataset, model, regime (in-domain or held out) and arm: the difference in R@5 points, arm minus SNK, of the
seed mean (three seeds, both splits where the dataset is read under both), with a 95% interval from 2,000 paired
bootstrap resamples over questions, the three seeds' per-question scores averaged. Call per cell: BELOW if the whole
interval is below zero, ABOVE if it is above, LEVEL otherwise. Beside each cell: the number of the six fits where the arm
is below SNK. Derived columns, from the same cells:

- what S adds = SNK - NK (all else in) and S - 0 (alone); what NER adds = SNK - SK (all else in); what knn adds =
  SNK - SN; what NER and knn add together beside S = SNK - S. Each is the difference of two arm cells, with the interval
  from the paired per-question differences, never from the two intervals;
- the GNN - MLP difference per arm (does message passing matter more when fewer families are in).

The stage decides nothing about adoption. It files the grid and the calls.

## 6. Cost and order

PKOH measured 80 to 90 lane-hours for the whole PKO build (11 carves); the six development carves are about two thirds of
it, and an arm with fewer families builds fewer rows. Estimate per arm: 45 to 60 lane-hours and 86 GiB of disk (look
32 GiB of it, regenerated per arm because the rows differ). The arms are therefore sequential on disk: build, read the
twelve fits, file, then prune the arm's look and cache (the reads and records stay). Smallest dataset first inside an arm
(webqsp, musique, hotpotqa, metaqa, squad, 2wiki), so a partial arm still yields cells. About 12 to 15 hours per
priority arm at 12 lanes; the five arms are about three days of host time, GPU reads (twelve per arm) beside the IDP fits.

## 7. Not claimed

- No refit: the numbers are the models of record on pools and edges with a family missing.
- The types arrays (EM and score classes) stay the record's, as in B1P's b1t: they are per-entry classes of the
  record's chosen graph and are not recomputed for an arm.
- An arm's number is a development-split read. Nothing here touches a test split.
