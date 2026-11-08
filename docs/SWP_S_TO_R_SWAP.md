# SWP: swap S for R with N and K fixed, and isolate what the structural family carries (same-input pair)

Declared 8 November 2026 (about 04:00 IST), before any IDPB read. Follows [CANONICAL_LINK_AND_POOL_LAYOUT.md](CANONICAL_LINK_AND_POOL_LAYOUT.md),
[B1H_MODELS_OF_RECORD_ON_HYPERLINKS.md](B1H_MODELS_OF_RECORD_ON_HYPERLINKS.md) (filed 22a00e5),
[IDP_IDENTICAL_INPUT_PAIR.md](IDP_IDENTICAL_INPUT_PAIR.md) (A1 filed 70ec6cb) and [B1P_PKO_CHAIN_ON_B1.md](B1P_PKO_CHAIN_ON_B1.md). Code:
`outputs/mp_unified/zidpb.py` (new; imports zidpa, zb1h, zb1m, zb1p, zgnx, zml1s, zp1c3 and edits none). Outputs only under `outputs/idpb/`.
Nothing trains. No filed pool, fit, read or record changes.

## 1. The question (the user, 8 November, in order)

- For this paper the families are S, N and K (all three exist as edges). Where S is removed and R (the rule-built links) put in its
  place, how much of the trained model's recall is retained? 2Wiki and HotpotQA are done.
- Make it uniform: the same swap on MuSiQue, WebQSP, MetaQA and SQuAD. Then the effect of a rule-built link against the dataset's
  own structural link is isolated, and so is whether message passing is needed.
- The models are the same-input pair (GNX and its equivalent MLP, zrckpn); "supersede everything else"; in-domain results with the
  same input. If there is no time to retrain, use the models directly, and cross-fit the older models trained on the SNK graph.
- The paper's wording must reflect this experiment and these goals.

## 2. What the records already hold (read 8 Nov from `outputs/b1h/report.md`, B1H, filed 22a00e5)

HippoRAG 2's three 1,000-question settings, in-domain, 3-seed means, R@5 x100. b1t: structural slot empty, N and K kept. b1w: the rule's
links (W2_S) in the slot. b1: the dataset's own S in the slot (2wiki, hotpotqa: the corpus hyperlinks; **musique: the package's
title-mention graph, which is itself a rule**).

| setting | model | b1t | b1w | b1 | b1w - b1t | b1 - b1w (95% interval) |
|---|---|---:|---:|---:|---|---|
| 2wiki | zg6r | 78.1 | 94.9 | 95.4 | +16.8 | +0.4 (-0.3, +1.1) |
| 2wiki | zrckpn | 77.0 | 94.1 | 94.4 | +17.1 | +0.3 (-0.4, +0.9) |
| hotpotqa | zg6r | 91.1 | 94.3 | 96.5 | +3.1 | +2.2 (+1.5, +2.9) |
| hotpotqa | zrckpn | 90.4 | 93.4 | 95.6 | +3.0 | +2.2 (+1.5, +2.9) |
| musique | zg6r | 66.0 | 69.0 | 68.6 | +3.0 | -0.4 (-0.9, +0.2) |
| musique | zrckpn | 64.5 | 67.4 | 67.3 | +2.9 | -0.1 (-0.6, +0.4) |

- **Direction.** These fits were trained with the rule's links in the slot (U2c's pools: R + K on the passage datasets). The b1 column is
  therefore S swapped in at read time, the reverse of the user's direction (trained on S, R swapped in). The user's direction on the
  passage datasets comes only from the older S-trained fits (section 5).
- **Reading.** 2Wiki: the rule's links are level with S (interval holds zero). HotpotQA: S is above the rule by 2.2 points (interval
  excludes zero). MuSiQue: level, and S there is a rule. A "share of the gain" ratio is not quoted for MuSiQue: it divides two small
  gains whose intervals overlap (the value 1.03 to 1.14 in B1H's table says "level", not "more than all").
- **What is missing for the same-input pair.** zg6r and zrckpn have the whole ladder on these three settings. GNX has b1t and b1w
  (IDPA) and **no b1**.

## 3. Part A, declared now: IDPB, GNX on b1

- **Reads.** GNX's six fits (`scr-gnx-nx-s{0,1,2}` and `-hp`), forked to `outputs/idpb/fits/h/<fit>` without their reads (models.pt
  sha-checked), each read on B1H's carve b1 under B1H's gate for its tag (`main`: 2wiki and hotpotqa; `musique`: musique), B1H's roots
  (`outputs/b1h/pools`, `look`, `cache`, `links`, `kt`) and B1H's column folder `outputs/b1h/ltm` (read only), by `zml1s.run_arm(read)`
  under `zgnx.on_gnx`, the same body as IDPA's read with B1H's pieces. Twelve reads of about 4 minutes at 0.29 of the card; one test read
  first (s0, main), the other eleven wait for it, then the report.
- **Report** (`zidpb.py report`, `outputs/idpb/report.{json,md}`): for each setting and domain (in-domain, zero-shot), R@5 of GNX, zrckpn and
  zg6r on b1t, b1w and b1 (GNX's b1t and b1w are IDPA's reads, zrckpn's and zg6r's are B1M's, B1P's and B1H's, all read only); the steps
  b1w - b1t, b1 - b1w and b1 - b1t with paired 95% intervals (2,000 resamples of the questions, seed 0); the share of the b1 gain at b1w
  where b1 - b1t > 0; GNX minus zrckpn on each carve with a label by its interval; and the change of that gap from b1t to b1w and to b1
  (paired by question). The last row is the message-passing reading: if the gap does not grow when structure is put in, message passing
  adds nothing the structure's inputs did not already give the MLP.
- **No threshold, no adoption call.** The stage reports differences and intervals. It decides nothing about any model.

### 3a. Amendment A1 (8 Nov 04:10, before any IDPB read is filed)

The first test read (`idpb-main-m0`, 04:01) failed on `FileNotFoundError outputs/b1h/cache/2wiki/b1/part_0of1/xc.npy`: the host cleaner removed
B1H's regenerable b1 look chunks and cache arrays on 7 Nov 16:49 (no queued item named them; 12 of the 17 arrays per cache part are gone, the small
ones and every record remain; B1H's pools, links, kt, columns and gate are intact). `zidpb.py regen --dataset D` (rows `idpb-regen-{2wiki,hotpotqa,musique}`,
2 cpu, 6 to 8 GB) sets the filed look and cache folders of the carve aside as `b1.filed`, runs B1H's own chain on B1H's root, checks that every
regenerated array equals the sha256 the filed cache record names (a difference stops the stage), and puts the filed records back so no sha that names them
changes. The reads depend on it. Nothing else of B1H is touched; the regenerated files are pinned by the queued reads.

### 3b. Part A filed (8 Nov 04:53 IST; `outputs/idpb/report.{md,json}`, 604f21c)

GNX, zrckpn and zg6r on B1H's b1t -> b1w -> b1, R@5 in points, in-domain, 3-seed means; the interval is on b1 - b1w (S in the slot against the rule links, both with N and K in). The fits were
trained with the rule links in the slot, so b1 is S swapped in at read time.

| setting | model | b1t | b1w | b1 | b1 - b1w | call |
|---|---|---:|---:|---:|---|---|
| 2wiki | GNX | 77.6 | 95.0 | 95.3 | +0.3 (-0.4, +1.0) | level |
| 2wiki | zrckpn | 77.0 | 94.1 | 94.4 | +0.3 (-0.4, +0.9) | level |
| hotpotqa | GNX | 91.0 | 94.1 | 96.5 | +2.4 (+1.7, +3.1) | S above R |
| hotpotqa | zrckpn | 90.4 | 93.4 | 95.6 | +2.2 (+1.5, +2.9) | S above R |
| musique | GNX | 66.2 | 69.1 | 69.0 | -0.1 (-0.6, +0.4) | level (S is a title-mention rule) |
| musique | zrckpn | 64.5 | 67.4 | 67.3 | -0.1 (-0.6, +0.4) | level |

GNX minus zrckpn is +0.6 to +0.9 on 2wiki and hotpotqa and +1.6 to +1.8 on musique in-domain, on every carve, and the change of that gap from b1t to b1w or b1 holds 0 in
every in-domain row: putting structure in does not make the GNN gain on the MLP there (the zero-shot musique row shrinks it, -1.7 to -2.9). The b1 rows of GNX are new reads;
the b1t, b1w rows of zrckpn and zg6r are B1M's, B1P's and B1H's. The share column of the report prints only where b1 - b1w is not level. Failures before it ran: the cleaner had
removed B1H's regenerable look, cache, links and kt files (3a); the report's first run failed on `None in (...)` with arrays (fixed b49b7da).

## 4. Part B, declared 8 Nov 04:20 (before any pair-B read is filed): SQuAD, MetaQA and WebQSP, the same-input pair

Facts that fix the design:
- The pools of record are R + N + K on SQuAD (rule graph in the slot) and S + N + K on MetaQA and WebQSP (the KB triples in the slot).
- ABL's arms with the slot set and N and K in are RK (ARM `("R", 1, 1)`); the other two cells of the swap were not ABL arms, so `zabl.py` gained
  `ARM_SWP` (commit 7b03029): **SNK** = `("S", 1, 1)` and **NK** = `("", 1, 1)`, beside the eight filed subsets, which are unchanged.
  `ARM_ALL = {**ARM, **ARM_SWP}`; `applies`, `dropped`, `on_arm`, `items` and `--arm` read it. The self-test asserts N and K are in every swap arm.
- MetaQA has no in-domain fit (held out under both splits); WebQSP has no held-out fit. "In-domain on all six" holds for five.

**Cells (full dev carve `s1eval`, every cell is built and read the same way):**

| dataset | slot S | slot R | slot empty |
|---|---|---|---|
| SQuAD | SNK (new build) | REC (the record) | NK (queued, `swp3-*`) |
| MetaQA | REC | RK (ABL's row, moved up) | NK (queued, `swp3-*`) |
| WebQSP | REC | RK (ABL's row, moved up) | NK (queued, `swp3-*`) |

The pair is GNX and zrckpn (6 fits each: seeds 0 to 2 on both WQ splits), the same fits as Part A. Nothing trains. `zswp.py` (commit b25504e) forks each
fit to `outputs/swp/fits/<arm>/<fit>` without its reads (models.pt sha-checked), builds LTM's candidate lists and next-hop columns on the arm's carve
from LTM's own fold fits (`ltmcols.lists/cols` with `load_folds` pointed at LTM's folder, as B1H does), and reads each fit under `zgnx.on_gnx` or
`zltm.on_ltm` on the arm's roots and gate (`zabl.arm_roots`; REC uses the record's roots and LTM's own columns). Queue rows `swp2-*`: one fork per arm, lists and
two column rows per new cell, twelve reads per cell, a test read of GNX and of zrckpn on the record's SQuAD carve first (the other reads wait for them), a
report. The prune rows of the three new cells wait for their twelve reads, because the reads need the cache.

**Report** (`zswp.py report`, `outputs/swp/report.{json,md}`): per dataset and regime, R@5 of each model in each slot; R minus S, R minus empty and
S minus empty with paired 95% intervals (ABL's `compare`, 2,000 resamples of the questions); GNX minus zrckpn per slot. A cell without its arm is
listed as missing, not filled. The direction is named in every row: the fits of record were trained with S in the slot on the KB datasets and R in the
slot on SQuAD, so the other slot is a swap at read time (S -> R on the KBs; R -> S on SQuAD).

**No threshold, no adoption call.** The NK builds (the empty slot, rows `swp3-*`, same twelve reads per cell) sit behind the RK WebQSP rows on purpose, so they cannot starve that build. ABL's doc records the arms as its amendment A5. SQuAD's S is the title-mention rule, so
there too the comparison is rule against rule.

## 5. Part C, declared 8 Nov 05:30 (the user: "you need to train with S in the fit and R outside as well ... do this for all datasets"): the pair trained on S

**Question.** The fits of record for 2Wiki, HotpotQA, MuSiQue and SQuAD were trained with R in the slot (U2c), so Parts A and B read S in at read time on those
datasets: the reverse of the user's direction. Part C trains the same-input pair with **S in the slot (S + N + K)** and reads it with S, R and nothing in the slot,
so both directions stand side by side. On MetaQA and WebQSP the fits of record already trained on S; the S-trained fits are new there too (they are trained
on all five training datasets together, so one fit per split has S in every slot).

**What runs (seed 0 on both WQ splits, as ABL's amendment A3; one fit per model and split).**
1. Arm **SNKF** (`outputs/mp_unified/zswpc.py`, on ABL's chain, not edited): the five training carves (squad, hotpotqa, 2wiki: `fit`; musique: `s1fit`; webqsp: `fit`)
   built with the slot holding S, N and K in, under `outputs/abl/SNKF`; its gate over the five.
2. LTM on SNKF (`ltmcols.py`, not edited): lists, the five fold fits per split, the cross-fitted columns of the training carves.
3. The fits, in a separate copy of the code tree on the host (the "S-world": its own `outputs/pko` = SNKF, its own `screen`, `ml1`, `ltm`), so that no fit, stamp or
   column of the record can be written by them: zspk (seed 0), zg6r (starts from that zspk), GNX (zg6r + the columns of step 2), zrckpn (CPU, the columns of step 2).
   The code is the record's, byte for byte (shas listed in the S-world's manifest).
4. Reads of the four S-trained fits (GNX, zrckpn, both splits) with the slot S / R / empty: SQuAD, MetaQA, WebQSP on Part B's arms (s1eval; REC / SNK / NK / RK),
   2Wiki, HotpotQA, MuSiQue on B1H's carves (b1 = S, b1w = R, b1t = empty). Columns of every read carve come from the S-trained scorer of the fit's split.

**Report** (`zswpc.py report`, after the reads): per dataset and regime (in-domain / held out), per model, R@5 at each slot; the paired interval of R minus S and of
empty minus S for each model (trained on S), printed beside the same rows of Parts A and B (trained on R); GNX minus zrckpn at each slot. A step gets a call by its
interval (above / level / below); no threshold, no adoption call; no ratio is quoted without its interval.

**What it can and cannot show.** It shows, for each family, how much of a model's recall survives when the slot is swapped, from either training family. It cannot
show a seed spread (one fit per split; the intervals are over questions). On MuSiQue and SQuAD S is a title-mention rule, so the comparison there is rule against
rule. "Do we need message passing" is read from the GNX − zrckpn gap at each slot, not from the swap.

**Guards.** The S-world is read-only toward the record: a fit named like a record fit exists only inside it. The record's `outputs/screen`, `outputs/pko`, `outputs/ltm`
are never a write target of a Part C process. Rows pin the S-world's folders so the host cleaner keeps them. The SNKF build is checked on the first dataset
(squad) before the other four are queued.

### 5a. Amendment (8 Nov 05:35, the user: B1L's LLM-edge comparison belongs in the MuSiQue table): a reference column, no new compute

The Part A and Part C reports (and Part B's, where the dataset is one B1L read) carry B1L's filed rows (`outputs/b1l/report.json`, 568bf08) as a **reference block** for
MuSiQue and 2Wiki: the record fits `zg6r` and `zrckpn` (trained with R, not GNX) read on b1t / b1w / b1o, in-domain and held out, with B1L's calls. Rules for the block:
- it is labelled as reference: the edges are H4o's LLM-built P edges (not part of the standalone method), the fits are the record's, the pair is not the same-input pair;
- b1o's entries are NULL-heavy (89.5% of MuSiQue's entries have the null score type) and the fits were fitted on W2_S-typed entries, so b1o − b1t is a lower bound on what a
  model can take from LLM edges (B1L's own caveat);
- the direction is named (rule links against LLM edges); b1w − b1o is quoted with its interval and call (in-domain MuSiQue NOT_LESS: `zg6r` GREATER, `zrckpn` SAME;
  held out LESS for both); never "the rule beats the LLM";
- a GNX read on b1o (the same-input pair on the LLM edges) is a new cell, not queued here: it needs LTM's lists and columns on b1o (B1L's, on the record scorer) and, for
  Part C's S-trained fits, the S-scorer's columns on b1o.
It does not test an upper bound for a multi-hop graph: the LLM edges are extracted from the passages, not drawn between a question's chain. That test would be an oracle graph
(gold passages of one question linked to each other, label-using, a diagnostic upper bound only, never a method); it is **not** declared or queued.

### 5b. Part D declared 8 Nov 05:40 (the user: "sure"): an oracle multi-hop graph on MuSiQue, an upper bound, queued after Part C's builds

**Question.** On MuSiQue S and R are both rules, and H4o's LLM edges (B1L) lift no more than the rules. None of the three is drawn between a question's chain. Would a graph that
**is** drawn between the chain's passages lift the models, and by how much? That bounds what any better multi-hop graph can give; it is a diagnostic and never a method
(the edges are built from the gold labels).

**Design (amended 8 Nov 06:00, before anything is built).** The oracle pairs are built by `oraclepairs.py` (laptop-tested on the real files) from the 1,000 questions' gold
supporting paragraphs and MuSiQue's own `question_decomposition`; a hop's paragraph is matched to B1a's node by (title, text), and the hop nodes of every question equal the golds B1a filed
(1,000 of 1,000, checked by the build):
- **ORC (chain):** an edge between a sub-question's paragraph and the paragraph of each earlier sub-question its text refers to ("#1", "#2": the answers it depends on). A sub-question
  that refers to none is a branch root (3hop2, 4hop2, 4hop3) and gets no edge; every question's hops form one connected tree (n−1 edges: 2hop 1, 3hop 2, 4hop 3). **1,090 distinct pairs.**
- **ORA (all gold pairs):** an edge between every two supporting paragraphs of a question (ORC is a subset). **1,750 distinct pairs.**
**The cells add the oracle to the rule graph instead of replacing it:** **b1c = W2_S (R) pairs ∪ ORC** and **b1q = R ∪ ORA**, N and K as in b1w, the oracle entries untyped (NULL) as b1o's.
Reason: oracle edges join gold passages only, so a graph made of them alone would let a node's degree reveal that it is gold; on top of R every node keeps edges, and the cell
asks the question that matters, "what do the right chain edges add to what the rules already find" (b1c − b1w, b1q − b1w, b1q − b1c), beside b1t, b1o. The lift is a **ceiling** on
what a better graph could give these models, not an estimate of any buildable one: the pairs use the labels, and a model that reads "linked to a high-ranked candidate" is helped by them
by construction. The oracle-alone cell (no R) is not built unless asked.

**How (B1L's machinery, `zb1l.py`, not edited).** A new stage `zb1g.py` (same shape as zb1l: `bench`, `settings`, `chain`, `gate`, `fork`, reads, `report`) with the pair file
replaced by the oracle pairs added to b1w's: carves **b1c** (R ∪ ORC) and **b1q** (R ∪ ORA) (named not to clash with B1a, B1P, B1H); folders `outputs/b1c/`, `outputs/b1q/`. Reads: the fits of record (`zg6r`, `zrckpn`: R-trained) and GNX, in-domain (WQ-hotpotqa fits) with the
held-out read beside, exactly B1L's report (b1c / b1q minus b1t, b1w, b1o, and b1q − b1c, paired by question, calls by the interval). Part C's S-trained pair is read on the same carves once Part C is built.

**Reading it.** The oracle lifts a lot over b1w and b1o: a better multi-hop graph can help and our rules are the limit on MuSiQue. It lifts about as little: the limit is the first stage
and the hop depth, not the graph. Either way the call is by the interval, no threshold; ORA ⊇ ORC by construction (more edges); ORC is the one to quote as "the chain edges", always as a ceiling.
**Not queued yet:** the stage `zb1g.py` (the pairs come from `oraclepairs.py`, which reads B1a's raw MuSiQue files, `outputs/bench/hipporag2/raw/`) and the two chains (two full pool chains each, as B1L's note) wait
for Part C's builds to finish so that they do not compete for the 26 cpus; the builder's selftest comes first.

## 6. What the paper's wording must carry (for the paper session; this stage edits nothing under paper/)

- Name the direction of every swap (trained on which family, read with which).
- "Links matter more than message passing" and "the rule recovers X%" need the b1t baseline named as a read-time shift and need the
  MuSiQue case stated as rule against rule.
- The claim the experiment supports, once Parts A and B are filed: how much of the dataset's own structure a rule-built link recovers for a
  model, and whether message passing adds beyond the same inputs. Nothing about "needing" message passing before the gap-change row is read.

## 7. Disclosure

Development reads on HippoRAG 2's 1,000-question settings (B1 carves), no test split, no LLM. Seeds 0 to 2 of the fits of record on both WQ
splits. The b1t, b1w rows of GNX come from IDPA's reads, unchanged.
