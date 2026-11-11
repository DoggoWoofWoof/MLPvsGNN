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

**Amendment (9 Nov 11:28 IST, before Part B is filed): the intervals.** ABL's `compare` averages each chunk of 100 bootstrap resamples into one
number, so its 2,000 resamples are 20 values and its intervals are about 10 times too narrow (found 8 Nov 18:45 on Part C's reads). Part B's
intervals and calls come from `zswpbr.py report` (`outputs/swp/report_pb.{json,md}`): zswp's report with Part C's per-resample comparison
(`zswpcr.cmp`, the same seed and resamples), swapped in-process because the S-world and the N-world pin zswp.py and zabl.py until their reads
end. zswp's own `report.md` keeps the narrow intervals and is not quoted.

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
**The S-world's modes (8 Nov 05:27).** `sworld.py mode train|read` repoints the S-world's `outputs/pko` to `outputs/abl/SNKF` (train: the fits are trained with S in the slot) or to the record's `outputs/pko` (read: the S-trained fits are read on the record's carves, Part B's arms and b1 / b1w / b1t). `run [--mode]` refuses a command whose mode differs from the file. Training goes through `zswpt.py` (train-only; the read of each fit is replaced by a no-op; it imports neither `zabl` nor `zswpc`, so zrckpn keeps zltm's own 6-thread, GPU-hidden environment and no module sets the PKO 2-thread one). `verify` fails only on a changed byte of the copied record code and prints a note for files added to the record after `make`.

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

### 5b. Part D declared 8 Nov 05:31 (the user: "sure"): an oracle multi-hop graph on MuSiQue, an upper bound, queued after Part C's builds

**Question.** On MuSiQue S and R are both rules, and H4o's LLM edges (B1L) lift no more than the rules. None of the three is drawn between a question's chain. Would a graph that
**is** drawn between the chain's passages lift the models, and by how much? That bounds what any better multi-hop graph can give; it is a diagnostic and never a method
(the edges are built from the gold labels).

**Design (amended 8 Nov 05:37, before anything is built).** The oracle pairs are built by `oraclepairs.py` (laptop-tested on the real files) from the 1,000 questions' gold
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

**Measured before any model (8 Nov 05:41, `oraclepairs.py coverage`, a count over the filed graph files of B1's MuSiQue setting, 11,656 passages, 1,000 questions).** Of the 1,090 chain
pairs (ORC), the graph already links: S (b1, title mentions) 294 (27.0%), R (b1w, W2_S) 309 (28.3%), N (NER) 629 (57.7%), K (kNN) 248 (22.8%); S+R+N+K together 764 (70.1%), R+N+K 762
(69.9%), S+N+K 758 (69.5%). Of the 1,750 all-gold pairs (ORA): S 17.0%, R 17.9%, N 38.0%, K 14.6%, union 46.1%. Each graph holds 26-75 thousand pairs, of which under 1% are chain pairs
(R 0.75%). So the rule links (and the dataset's own title-mention links) hold about a quarter of the chain's edges, NER holds more than half, and 30% of the chain's edges are in no family;
the cells b1c / b1q add the 781 / 1,437 pairs R does not hold. This is a statement about the graphs, not about a model: it does not say how much a model gains from an edge it is given.

**How (B1L's machinery, `zb1l.py`, not edited).** A new stage `zb1g.py` (same shape as zb1l: `bench`, `settings`, `chain`, `gate`, `fork`, reads, `report`) with the pair file
replaced by the oracle pairs added to b1w's: carves **b1c** (R ∪ ORC) and **b1q** (R ∪ ORA) (named not to clash with B1a, B1P, B1H); folders `outputs/b1c/`, `outputs/b1q/`. Reads: the fits of record (`zg6r`, `zrckpn`: R-trained) and GNX, in-domain (WQ-hotpotqa fits) with the
held-out read beside, exactly B1L's report (b1c / b1q minus b1t, b1w, b1o, and b1q − b1c, paired by question, calls by the interval). Part C's S-trained pair is read on the same carves once Part C is built.

**Reading it.** The oracle lifts a lot over b1w and b1o: a better multi-hop graph can help and our rules are the limit on MuSiQue. It lifts about as little: the limit is the first stage
and the hop depth, not the graph. Either way the call is by the interval, no threshold; ORA ⊇ ORC by construction (more edges); ORC is the one to quote as "the chain edges", always as a ceiling.
**Built and queued (8 Nov 05:43).** `zb1g.py --variant c|q` (selftest OK for both; `bench` built `outputs/b1c/` and `outputs/b1q/` on the laptop in under a second each: 84,420 and 85,732 directed structural edges against b1w's 82,858) and 40 rows `b1gc-*` / `b1gq-*` (B1L's MuSiQue rows mirrored: selftest, settings, fork, chain, gate, lists, columns for both splits, and the twelve reads of the record's `zg6r` and `zrckpn`, seeds 0-2). They sit after Part C's rows in the file, so Part C's builds have the priority. Still to add after Part C: the GNX reads on b1c / b1q, the S-trained pair on the same carves, and the report (b1c / b1q against b1t, b1w, b1o).

### 5c. Part D filed (8 Nov 13:53 IST; `outputs/b1g/report.{md,json}`, `calls.json`): the oracle chain edges lift both models by about 11 points in-domain, as a ceiling

All 40 rows `b1gc-*` / `b1gq-*` finished rc 0 on the host; the 24 read files were fetched (`rx fetch --glob`) and `zb1g.py report` ran on the laptop (missing reads 0). In-domain R@5 x100 on
MuSiQue's 1,000 questions, no new fit, the models of record read on four link sets:

| model | b1t | b1w (R) | b1c (R + ORC) | b1q (R + ORA) | b1c - b1w | b1q - b1w | b1q - b1c |
|---|---:|---:|---:|---:|---|---|---|
| zg6r | 66.0 | 69.0 | 79.6 | 80.3 | +10.7 (+9.6, +11.7) | +11.3 (+10.2, +12.5) | +0.6 (+0.1, +1.2) |
| zrckpn | 64.5 | 67.4 | 78.2 | 78.9 | +10.8 (+9.7, +11.8) | +11.4 (+10.4, +12.6) | +0.7 (+0.1, +1.2) |

All six pairs (b1c-b1w, b1q-b1w, b1q-b1c, b1c-b1t, b1q-b1t, b1w-b1t) are UP for both models by the interval. By hop depth (b1c - b1w): 2-hop +7.6 / +8.8, 3-hop +15.3 / +13.1, 4-hop
+11.5 / +12.6 (zg6r / zrckpn); ORA adds beyond ORC only on 4-hop questions (+5.4 / +4.9, UP), FLAT on 2-hop and 3-hop. The fits that did not train on MuSiQue: b1c 77.0 / 77.0, b1q 82.1 / 81.0.
B1L's LLM edges (b1o, filed, across stages) read 67.4 / 67.3 in-domain: no lift over b1w, against about 11 points for the oracle edges.

**Reading it (the branch declared above: "the oracle lifts a lot").** For these models, edges drawn between a question's own supporting paragraphs lift R@5 far more than anything the
rules, the dataset's links or the LLM edges give, so on MuSiQue the graph is a limit beside the first stage. What the table does **not** say: (1) it is a ceiling that uses the labels, never an
estimate of a buildable graph; R already holds 28.3% of the chain pairs and the other 781 are what b1c adds; (2) oracle edges join gold paragraphs only, so even on top of R a node that gains
edges is gold, and a model that reads "linked to a high-ranked candidate" is helped by construction (the held-out fits, which learned other graphs, gain more: b1q - b1c +5.0 / +4.0).
**The control that separates a connection from a degree cue is not built yet:** decoy edges with the same sources and a random non-gold corpus node as the target (same number of pairs, same
degree gain on the golds, no gold-gold connection). If the decoy lift is near zero the +11 is the connection; if it is large, most of it is the degree cue. Proposed as Part D2, same
machinery (`zb1g.py` with a third kind), about 20 rows; not queued until built and declared here.

### 5d. Part D2 declared 8 Nov 14:54 (before any control read is filed): the two controls for the oracle edges, b1d (decoys) and b1e (swapped chains)

**Question.** Part D (5c) lifted both models by about 11 points when the chain's edges were added to R. Two things come with those edges besides the connection between the right
passages: every node they touch is gold for some question, so a gold node gains degree, and a model that reads "linked to a high-ranked candidate" can use that. D2 builds two graphs
from the chain edges with the labels but without the right connection and reads the same models on them. It is a diagnostic, never a method.

**The controls (built by `oraclepairs.py`, seed 0, both seeded and rebuilt equal; ORC and ORA unchanged, sha16 `0457f765…` and `e28c564a…`, 1,090 and 1,750 pairs).**
- **OXC, the degree control (carve b1d = R ∪ OXC).** Every node ORC touches (1,456 golds) keeps exactly its ORC degree, but each of its pairs goes to a node drawn uniformly (seed 0) from the corpus
  nodes that are gold for no question of the 1,000: **2,180 pairs**, each with one gold end and one non-gold end, 1,968 distinct decoy nodes (at most 3 pairs each); no pair joins two golds. 2 of
  the 2,180 are already in W2_S. sha16 `db7702e0…`. If the lift of b1d over b1w is near zero, the +11 of b1c is not a degree gain on the golds; if it is large, it is.
- **OXG, the connection control (carve b1e = R ∪ OXG).** ORC's 1,090 pairs with their far ends exchanged between questions: every node keeps its ORC degree and the pair count stays
  1,090, but each new pair joins gold paragraphs of two different questions and is in neither ORC nor ORA (overlap 0 and 0); 1 of 1,090 is already in W2_S. sha16 `33bd08c5…`. It keeps
  everything the chain edges have except the right partner: gold nodes with the same degree and gold-to-gold links. If b1e lifts as much as b1c, the lift is "gold nodes linked to
  gold nodes", not the chain; if it lifts less, the right partner is what the models use.
Neither control is a buildable graph (both use the labels), and neither is quoted as an estimate of one.

**Cells and calls.** Same machinery as Part D: `zb1g.py --variant d|e` (selftest OK for c, q, d, e; `bench` built `outputs/b1d/` and `outputs/b1e/`, 87,214 and 85,036 directed structural
edges against b1w's 82,858), the same 20-row pattern as `b1gc-*` / `b1gq-*` per carve (`b1gd-*`, `b1ge-*`, 40 rows in all: selftest, settings, fork, chain, gate, lists, columns for both splits, and the twelve reads of
the models of record `zg6r`, `zrckpn`, seeds 0-2 on both splits), queued in the file right after Part D's block. In-domain = the mean over the WQ-hotpotqa fits, as in 5c, with the held-out read beside; paired by question, 95% bootstrap (2,000
resamples, seed 0), calls UP / DOWN / FLAT by the interval, no threshold. The pairs `zb1g.py report-d2` files (to `outputs/b1g2/`): b1c-b1w (as 5c), **b1d-b1w, b1e-b1w, b1c-b1d, b1c-b1e,
b1d-b1e**, overall, by hop depth (2/3/4) and held-out. Nothing is tuned and no fit is trained.

**Reading it (branches fixed now).** (i) b1d-b1w FLAT and b1e-b1w FLAT or below b1c: the lift is the connection between the right passages. (ii) b1d-b1w UP: part of the lift is a degree gain on
the golds, and the share is read from b1c-b1d, always as a difference in points, never as a ratio. (iii) b1e-b1w UP with b1c-b1e FLAT: the lift is gold-to-gold linking, not the chain.
(iv) b1c-b1d and b1c-b1e both UP with the controls UP too: both parts exist, said so with the points. Whatever comes out, the oracle numbers stay a ceiling for the paper. The 40 rows are
the "about 20 rows" of 5c, per carve.

### 5e. Part D2 filed (8 Nov 17:00 IST; `outputs/b1g2/report.{md,json}`, `calls.json`): neither control lifts; the +11 of the oracle chain edges is the right connection, not a degree cue

All 40 rows `b1gd-*` / `b1ge-*` finished rc 0 on the host; the 24 read files were fetched (`rx fetch --glob`) and `zb1g.py report-d2` ran on the laptop (missing reads 0). In-domain R@5 x100 on
MuSiQue's 1,000 questions, the models of record, no new fit, paired by question (95% bootstrap, calls by the interval):

| model | b1w (R) | b1c (R + ORC) | b1d (R + OXC, decoys) | b1e (R + OXG, swapped) | b1d - b1w | b1e - b1w | b1c - b1d | b1c - b1e |
|---|---:|---:|---:|---:|---|---|---|---|
| zg6r | 69.0 | 79.6 | 67.5 | 67.2 | -1.5 (-2.0, -0.9) | -1.7 (-2.2, -1.2) | +12.1 (+11.0, +13.3) | +12.4 (+11.3, +13.5) |
| zrckpn | 67.4 | 78.2 | 66.4 | 66.4 | -1.0 (-1.5, -0.5) | -1.0 (-1.5, -0.5) | +11.8 (+10.7, +12.9) | +11.8 (+10.7, +13.0) |

Calls (both models): b1c-b1w UP, **b1d-b1w DOWN, b1e-b1w DOWN**, b1c-b1d UP, b1c-b1e UP, b1d-b1e FLAT. The fits that did not train on MuSiQue: b1c - b1w +16.3 / +15.2 (zg6r / zrckpn),
b1d - b1w -0.3 (-1.0, +0.4) / +0.0 (-0.7, +0.7), b1e - b1w -0.9 (-1.6, -0.3) / +0.3 (-0.4, +0.9). By hop depth the controls' point estimates sit below b1w at every depth, with intervals clear of 0 on 3-hop and 4-hop (on 2-hop they touch 0)
(b1d - b1w -1.9 / -1.2 and -2.7 / -1.8; b1e - b1w -2.5 / -1.3 and -2.3 / -2.0), and b1c - b1d, b1c - b1e are UP at every depth.

**Reading it (declared branch (i), with one deviation).** Giving the golds the same extra degree (b1d) or linking golds to golds of other questions (b1e) does not lift either model: the +11
points of the oracle chain edges (5c) come from the edge joining the right passages of the question, not from a node's degree gain, from "linked to a gold" or from any extra edges. The
deviation from the wording of branch (i): I wrote the controls would be FLAT; in-domain they are **DOWN by 1.0 to 1.7 points** (the interval excludes 0), held-out about 0, so wrong-partner
edges cost a little in-domain and the models are not fooled by them. Nothing here says a buildable graph reaches the +11: the oracle is still a ceiling that uses the labels, and the controls
are not graphs anyone can build either. The share of the oracle's lift that a degree cue explains is therefore about zero in points (b1d - b1w <= 0); it is not quoted as a ratio.
The S-trained pair is not read on b1c / b1d / b1e (Part C, still building); GNX is not read on them.

### 5f. Part E declared 9 Nov 11:06 IST (before any Part E number): the pair trained with the slot EMPTY, read with nothing, R and S

The user, 9 Nov: "there is a difference between training without the link itself and with the original link, when we swap we are seeing how much it can transfer
without the exact data which is like generalization, but if we dont train itself with that slot the model wont know what to do with it so how does it answer our
question", then "if it comes back run your experiments".

**Question.** Every empty-slot cell so far (b1t; Part B's NK) reads a fit that was trained with R or S in the slot: it is a removal cost at read time, a transfer
measure, not what a model trained without the slot reaches. Part E trains the same-input pair with the slot empty (N + K only) and reads it with nothing, R and S in
the slot. The diagonal (each model trained and read with the same slot: R-trained/R = the record, S-trained/S = Part C, N-trained/empty = Part E) answers the
guide's point 4: what the links add when the model is trained to use them.

**What runs (seed 0 on both WQ splits, as Part C; Part C's steps with NKF in place of SNKF and a third tree, the N-world, in place of the S-world; no record or
Part C module is edited).**
1. Arm **NKF** (`outputs/mp_unified/zswpe.py`, on ABL's chain, not edited): the five training carves (squad, hotpotqa, 2wiki: `fit`; musique: `s1fit`; webqsp:
   `fit`) built with the slot empty and N and K in (ABL's NK arm, `("", 1, 1)`), under `outputs/abl/NKF`; its gate over the five.
2. The **N-world** (`outputs/host_ops/nworld.py`): `sworld.py`'s make / verify / mode / run on a sibling tree `../nworld` whose `outputs/pko` is NKF in train mode
   and the record's PKO base in read mode. `sworld.py` is not edited (the S-world pins it); runs carry `NWORLD=1`, and the Part E modules refuse any other tree.
3. LTM on NKF (`ltmcols.py lists / fit / cols`, not edited) in the N-world: the NK-trained scorer's fold fits and the columns of the training carves.
4. The fits (`zswpe_t.py`: `zswpt.py`'s train-only step with the N-world's checks): zspk (seed 0), zg6r (from that zspk), GNX (zg6r + the columns of step 3),
   zrckpn (CPU, the columns of step 3), both splits; GPU shares as Part C's final rows.
5. The reads (`zswpe_r.py`: `zswpr.py`'s engine with the N-world's checks) of the four N-trained fits (GNX and zrckpn, both splits) on Part C's slots:
   MuSiQue, 2Wiki, HotpotQA on b1t / b1w / b1; SQuAD on NK / REC / SNK; MetaQA and WebQSP on NK / RK / REC. The columns of every read carve come from the
   N-trained scorer of the fit's split. `swpe_collect.py` copies the reads to `outputs/swpe/fits`.

**Report** (`zswpe_rep.py`, laptop, after the reads; Part C's corrected per-resample bootstrap, 2,000 resamples, never `zabl.compare`): per dataset, regime
(in-domain / held out) and model, R@5 in each slot for the three training worlds (N-trained: Part E; R-trained: Parts A and B; S-trained: Part C), and the paired
intervals of:
- (a) **the diagonal steps**: R-trained read with R minus N-trained read empty, and S-trained read with S minus N-trained read empty (what a slot adds when the
  model trains with it);
- (b) **the training effect at the empty slot**: N-trained minus R-trained, both read empty (how far the read-time removal is from training without the slot);
- (c) **message passing**: GNX minus zrckpn in every N-trained cell, and the gap change between the diagonal cells (R-trained/R against N-trained/empty);
- (d) beside, no claim: the N-trained fits read with R and with S (transfer to a family the fit never trained with).
A step gets a call by its interval (above / level / below), with the direction of the subtraction named; no threshold, no adoption call; no ratio without its
interval. If Part C has not filed when Part E's reads are in, the report carries the R-trained and N-trained worlds and lists the S-trained cells as missing.

**What it can and cannot show.** It shows how much of each model's recall depends on the slot when training matches the input. It cannot show a seed spread (one
fit per split; the intervals are over questions). "Without links" means without the structural slot only: N (rule NER links) and K (kNN) stay in every cell,
so GNX still passes messages over N + K. On MuSiQue and SQuAD S is a title-mention rule, so S against R there is rule against rule.

**Order and cost.** Part C's remaining rows go first (the S-world's GPU fits and its 72 reads). NKF's five chains run on the CPU beside them (SNKF took about
1 h per carve and about 10 h for MuSiQue, the critical path), then about 10 h of GPU (zspk, zg6r and GNX on both splits, then the reads) and the two zrckpn CPU
fits; 179 rows (42 on the GPU), generated from Part C's last working rows by `outputs/host_ops/swpe_rows.py` (names `swpe-*`). The six read
cells that zabl.py builds and prunes (SNK on SQuAD, NK on SQuAD, MetaQA and WebQSP, RK on MetaQA and WebQSP) lose their look and cache to their prune
rows once their own reads are filed (SNK/squad and RK/metaqa were rebuilt for Part C on 8 Nov); each gets a resume of zabl's chain and its gate
(`swpe-build-*`, `swpe-gate-*`: seconds when nothing was pruned) after Part C's collect row and, for RK and SNK, after the cell's prune row; NK's
prune rows end rc 1 without removing anything (`zabl.prune` needs ABL's own reads of the arm, which NK does not have), so NK's rebuild waits on Part
C's collect alone. Part E's lists, columns and reads of the cell wait on that gate.

**Guards.** As Part C: the N-world is read-only toward the record and the S-world; a fit named like a record fit exists only inside it; every Part E step
refuses outside the N-world or in the wrong mode; no row names a directory as an input (the 8 Nov feeder crash). No Part E number goes into the paper before
this section records the filed report; section 6 applies to every Part E sentence.

**Amendment A2 (11 Nov 06:19 IST, before any Part E number): the start encoder of the slot-empty build.**
- What failed: both zg6r fits stopped at their start on 11 Nov 05:44-05:56. Each ran twice and ended rc 1 in 3 s; nothing trained. The feeder dropped the 89 rows
  behind them: both GNX fits, the flip, the seven forks, the 72 reads, the SNK/squad and RK/metaqa lists and columns, and swpe-collect.
- Why: `zml1s.load_encoder` takes ML1's zspk start only when its encoder holds `zg6rw.ENC_ELEMENTS` = 224,737 numbers, the base build's block encoder at
  hidden 128. Part E's zspk fits recorded the blocks `DISTS` and `depth_STRUCT` as dead (`lean_gpu.dead_blocks` leaves out every block whose values do not vary
  over the training rows, and the slot's two blocks are constant when the slot is empty). Their input set has 7 blocks, not 9, and the start's `l1.weight` is
  (128, 243), not (128, 309): 66 columns fewer, 216,289 numbers. Every other encoder tensor has the base shape. This holds on both splits; the fits'
  `models.pt` were read on the host. GNX, under ISO's KEEP, loads the same start through `ml1start.LOAD`.
- Fix: the zg6r and GNX fits run through `outputs/mp_unified/zswpe_tn.py` (new). It is `zswpe_t.py` unchanged, except that each start load takes 216,289 for
  the call and requires `l1.weight` (128, 243). `zg6r.copy_encoder` still checks every name and shape of the start against the model it builds. No record,
  Part C or Part E module is edited. The N-world is made again so that it holds the new file. The failed fits and the dropped rows are requeued as -r2 / -r3
  copies.
- Reading (c) and (d): a fit that left the two blocks out has no input weights for them. With R or S in the slot, those two blocks do not enter it; what the read's
  links feed through the other parts (chain match, KT, the other blocks and, for GNX, message passing) still does. (a) and (b) read the N-trained fits with the
  slot empty, where the two blocks are constant anyway.

### 5g. Part C filed (11 Nov 06:33 IST; `outputs/swpc/report.{md,json}`, 79faec1): from either training family, R is level with S on 2Wiki and in-domain MuSiQue, below S on HotpotQA and far below S on the KBs

All of Part C's reads and `swpc8-collect-r2` ended rc 0 on the host (collect 11 Nov 05:44 IST). The 72 S-trained reads were fetched from `outputs/swpc/fits` (`rx fetch --glob`)
and read beside the 72 reads of the fits of record (Parts A and B, B1H, B1M) by `zswpcr.py report` on the laptop: missing 0, 2,000 resamples, each the mean of its own
question draw (not `zabl.compare`; see section 4's amendment of 9 Nov). R@5 in points, seed 0, one fit per model and split, calls by the 95% interval over the questions.
"S-trained" = S in the slot of every training carve (SNKF). "R-trained" = the fits of record: R in the slot of the passage datasets' training carves (U2c); WebQSP's
training carve holds S in both families, so on WebQSP the two families differ only in the other carves. The swap read with each family's in-domain fits (MetaQA: held out):

| dataset | model | S-trained: S / R / none | R - S | R-trained: S / R / none | R - S |
|---|---|---|---|---|---|
| MuSiQue | GNX | 69.3 / 69.0 / 66.3 | -0.3 (-1.1, +0.4) level | 69.3 / 69.5 / 66.8 | +0.2 (-0.5, +0.8) level |
| MuSiQue | zrckpn | 67.3 / 67.6 / 63.9 | +0.4 (-0.3, +1.1) level | 67.4 / 67.8 / 64.6 | +0.4 (-0.4, +1.1) level |
| 2Wiki | GNX | 95.7 / 95.2 / 77.9 | -0.4 (-1.1, +0.3) level | 95.3 / 95.0 / 77.5 | -0.3 (-1.0, +0.4) level |
| 2Wiki | zrckpn | 94.7 / 94.3 / 77.1 | -0.4 (-1.1, +0.3) level | 94.4 / 94.2 / 77.4 | -0.2 (-0.9, +0.5) level |
| HotpotQA | GNX | 96.9 / 94.0 / 89.8 | -2.9 (-3.7, -2.1) R below S | 96.9 / 94.2 / 91.1 | -2.6 (-3.5, -1.9) R below S |
| HotpotQA | zrckpn | 96.3 / 93.5 / 90.0 | -2.9 (-3.7, -2.1) R below S | 95.7 / 93.5 / 90.3 | -2.2 (-2.9, -1.5) R below S |
| SQuAD | GNX | 91.1 / 91.1 / 91.2 | -0.0 (-0.2, +0.1) level | 91.0 / 91.5 / 91.0 | +0.5 (+0.3, +0.7) R above S |
| SQuAD | zrckpn | 91.2 / 91.1 / 91.3 | -0.1 (-0.2, +0.1) level | 91.1 / 91.3 / 91.3 | +0.3 (+0.1, +0.4) R above S |
| MetaQA (held out) | GNX | 35.6 / 2.3 / 0.5 | -33.3 (-34.1, -32.5) R below S | 37.5 / 9.7 / 0.6 | -27.8 (-28.5, -27.1) R below S |
| MetaQA (held out) | zrckpn | 38.0 / 1.1 / 0.7 | -36.8 (-37.7, -36.1) R below S | 39.0 / 1.3 / 0.7 | -37.7 (-38.5, -36.9) R below S |
| WebQSP | GNX | 66.6 / 11.4 / 8.3 | -55.2 (-57.3, -53.0) R below S | 66.4 / 10.6 / 7.8 | -55.7 (-57.8, -53.5) R below S |
| WebQSP | zrckpn | 64.9 / 10.4 / 7.8 | -54.4 (-56.6, -52.1) R below S | 64.6 / 10.6 / 8.2 | -54.0 (-56.2, -51.7) R below S |

**Reading it.**
- The swap, from either training family. Every in-domain R - S call has the same direction in both families except SQuAD (level when S-trained; R above S by 0.3-0.5
  when R-trained, rule against rule). So the R-trained fits of Parts A and B are not what made R level with S on 2Wiki and in-domain MuSiQue: fits trained with S in
  the slot, read with R, are level with S there too (-0.4 to +0.4). On HotpotQA R stays below S whichever family trained the fits (in-domain -2.9 / -2.9 S-trained,
  -2.6 / -2.2 R-trained; held out -2.1 / -2.1 and -1.8 / -1.2). On the KBs R does not stand in for S from either family (R@5 1.1 to 11.4 against 35.6 to 66.6).
  MuSiQue held out: R above S by +1.9 for S-trained zrckpn and for R-trained GNX, level for the other two.
- The training family at a fixed read slot (the report's second table, S-trained minus R-trained) is level in 32 of its 48 cells. The cells 1 point or more away from 0:
  MuSiQue held out (GNX read with R -1.3; zrckpn read with R +2.0, with nothing +2.2), HotpotQA in-domain GNX read with nothing -1.3, MetaQA read with S
  (GNX -1.8, zrckpn -1.0) and MetaQA GNX read with R (-7.4).
- Message passing on the same inputs (GNX - zrckpn, the report's third table): in-domain GNX is above zrckpn in all 6 MuSiQue cells (both families, every slot,
  +1.4 to +2.4) and in 5 of 6 on 2Wiki (+0.8 to +1.0; the R-trained empty slot is level), above in 3 of 6 on HotpotQA and on WebQSP (level in the rest), level in 5 of 6
  on SQuAD (R-trained empty slot -0.3); held out it is mostly level (MuSiQue 4 of 6, HotpotQA 5 of 6), and on MetaQA GNX is below zrckpn with S or nothing in the
  slot and above it with R. The change of the gap between slots (R vs S, S vs none, R vs none) is level in 33 of the 48 cells of `report.json`.
- Not shown: a seed spread (seed 0, one fit per model and split; the intervals are over questions). On MuSiQue and SQuAD S is itself a title-mention rule, so the
  swap there is rule against rule. The S-trained pair is not read on Part D's carves (b1c, b1d, b1e). The report's reference block (B1L's LLM edges, b1o, zrckpn
  R-trained only) is unchanged and says nothing about which graph is better.

### 5h. Part E filed (11 Nov 11:04 IST; `outputs/swpe/report.{md,json}`, b6705c2): trained with it, R or S in the slot adds +2.5 to +17.5 points on MuSiQue, 2Wiki and HotpotQA and S adds +34.7 to +39.3 on the KBs; with the slot empty GNX stays above zrckpn in-domain on MuSiQue and 2Wiki, and the change of that gap is level in every in-domain cell

All 72 of Part E's reads and `swpe-collect-r3` ended rc 0 on the host (collect 11 Nov 10:28 IST). The reads were fetched from `outputs/swpe/fits` (`rx fetch --glob`) and read
beside the 72 R-trained reads (Parts A and B) and the 72 S-trained reads (Part C) by `zswpe_rep.py report` on the laptop: missing 0, 2,000 resamples, each the mean of its own
question draw. R@5 in points, seed 0, one fit per model and split; in-domain reads the fit whose training split holds the dataset and held out the other (2Wiki, SQuAD and
WebQSP read both fits in-domain, MetaQA both held out); calls by the 95% interval over the questions. "N-trained" = the slot empty in every training carve (NKF); "empty" = no
structural slot, with N and K in every cell. The read slots (empty / R / S): b1t / b1w / b1 on MuSiQue, 2Wiki and HotpotQA; NK / REC / SNK on SQuAD; NK / RK / REC on MetaQA
and WebQSP. As in 5g, the R-trained fits held S in WebQSP's training carve.

| dataset | model | read: N-trained empty / R-trained R / S-trained S | R-trained R - N-trained empty | S-trained S - N-trained empty | N-trained empty - R-trained empty |
|---|---|---|---|---|---|
| MuSiQue (in-domain) | GNX | 66.5 / 69.5 / 69.3 | +2.9 (+1.6, +4.2) R-trained R above | +2.8 (+1.5, +4.1) S-trained S above | -0.3 (-1.3, +0.8) level |
| MuSiQue (in-domain) | zrckpn | 64.8 / 67.8 / 67.3 | +3.0 (+1.8, +4.2) R-trained R above | +2.5 (+1.3, +3.7) S-trained S above | +0.2 (-0.9, +1.4) level |
| MuSiQue (held out) | GNX | 50.8 / 62.9 / 60.9 | +12.1 (+10.3, +14.0) R-trained R above | +10.1 (+8.3, +12.1) S-trained S above | -11.2 (-13.0, -9.5) N-trained below |
| MuSiQue (held out) | zrckpn | 57.7 / 60.4 / 60.6 | +2.7 (+0.9, +4.4) R-trained R above | +2.9 (+1.1, +4.6) S-trained S above | -1.3 (-3.0, +0.3) level |
| 2Wiki | GNX | 78.1 / 95.0 / 95.7 | +16.9 (+15.5, +18.2) R-trained R above | +17.5 (+16.1, +18.8) S-trained S above | +0.6 (+0.0, +1.2) N-trained above |
| 2Wiki | zrckpn | 77.3 / 94.2 / 94.7 | +16.9 (+15.6, +18.3) R-trained R above | +17.4 (+15.9, +18.7) S-trained S above | -0.1 (-0.7, +0.6) level |
| HotpotQA (in-domain) | GNX | 90.3 / 94.2 / 96.9 | +4.0 (+2.9, +5.0) R-trained R above | +6.6 (+5.4, +7.8) S-trained S above | -0.8 (-1.7, +0.1) level |
| HotpotQA (in-domain) | zrckpn | 90.1 / 93.5 / 96.3 | +3.4 (+2.4, +4.4) R-trained R above | +6.2 (+5.1, +7.2) S-trained S above | -0.2 (-1.0, +0.5) level |
| HotpotQA (held out) | GNX | 89.3 / 93.3 / 95.2 | +4.0 (+2.9, +5.1) R-trained R above | +5.9 (+4.8, +7.0) S-trained S above | +0.5 (-0.5, +1.5) level |
| HotpotQA (held out) | zrckpn | 89.5 / 92.8 / 94.8 | +3.2 (+2.1, +4.2) R-trained R above | +5.3 (+4.2, +6.4) S-trained S above | +0.3 (-0.6, +1.2) level |
| SQuAD | GNX | 91.3 / 91.5 / 91.1 | +0.2 (+0.0, +0.4) R-trained R above | -0.2 (-0.4, -0.0) S-trained S below | +0.2 (+0.1, +0.4) N-trained above |
| SQuAD | zrckpn | 91.3 / 91.4† / 91.2 | +0.0 (-0.1, +0.2) level | -0.2 (-0.3, +0.0) level | +0.0 (-0.1, +0.2) level |
| MetaQA (held out) | GNX | 1.0† / 9.7 / 35.6 | +8.7 (+8.3, +9.1) R-trained R above | +34.7 (+33.9, +35.5) S-trained S above | +0.4 (+0.3, +0.5) N-trained above |
| MetaQA (held out) | zrckpn | 1.0 / 1.3 / 37.9† | +0.2 (+0.1, +0.4) R-trained R above | +36.9 (+36.1, +37.7) S-trained S above | +0.4 (+0.3, +0.5) N-trained above |
| WebQSP | GNX | 27.4 / 10.6 / 66.6 | -16.7 (-18.6, -14.8) R-trained R below | +39.3 (+37.0, +41.5) S-trained S above | +19.5 (+17.8, +21.2) N-trained above |
| WebQSP | zrckpn | 26.8 / 10.6 / 64.9 | -16.1 (-17.9, -14.3) R-trained R below | +38.1 (+35.6, +40.5) S-trained S above | +18.5 (+16.8, +20.2) N-trained above |

† `report.md` rounds each read twice (to 4 decimals in `report.json`'s cells, then to 1), so five of its reads sit 0.1 from the nearest value. They are given here from the
full-precision means (`report.json`, the a and b of the gap entries): SQuAD zrckpn R-trained R 91.4 (the report shows 91.3), MetaQA GNX N-trained empty 1.0 (0.9), MetaQA GNX
R-trained S 37.4 (37.5), MetaQA zrckpn S-trained empty 0.6 (0.7) and S-trained S 37.9 (38.0). Four of them are also in 5g's table: read its SQuAD zrckpn R-trained R as 91.4,
MetaQA GNX R-trained S as 37.4, and MetaQA zrckpn S-trained S as 37.9 and S-trained none as 0.6. Every difference and interval is computed at full precision and is
unaffected; reads at an exact tie (.x5) are shown as the report shows them. Take reads from `report.json`'s a and b, not from the reads tables.

**Reading it.**
- **The diagonal (5f's (a): what the slot adds when the model trains with it).** Trained and read with R, the pair is above the N-trained pair read empty in 13 of 16 rows;
  trained and read with S, in 14 of 16. Every MuSiQue, 2Wiki and HotpotQA row is above for both slots: in-domain MuSiQue +2.5 to +3.0, 2Wiki +16.9 to +17.5, HotpotQA (both
  regimes) R +3.2 to +4.0 and S +5.3 to +6.6, held-out MuSiQue +2.7 (R) / +2.9 (S) for zrckpn and +12.1 / +10.1 for GNX. On the KBs S adds +34.7 to +39.3, and R adds +8.7
  (GNX) and +0.2 (zrckpn) on MetaQA. On WebQSP R is below (-16.7 / -16.1), but that is not a diagonal step: the R-trained family trained with S on WebQSP's carve. On SQuAD
  every step is under half a point (GNX: R +0.2 above, S -0.2 below; zrckpn level); S there is itself a title-mention rule.
- **The training effect at the empty slot (5f's (b): N-trained minus R-trained, both read empty).** Level in 9 of 16 rows: in-domain MuSiQue, HotpotQA in both regimes,
  and zrckpn on 2Wiki, SQuAD and held-out MuSiQue. Where it is level, a fit trained with R and read with the slot removed (b1t; NK on SQuAD) stands in for a fit trained
  without the slot. N-trained is above on WebQSP (+19.5 / +18.5: read empty, the fits that trained with S on WebQSP's carve sit far below the fits trained without the
  slot), on MetaQA (+0.4 for both) and for GNX on 2Wiki (+0.6) and SQuAD (+0.2). It is below for GNX on held-out MuSiQue (-11.2).
- **Message passing on the same inputs (5f's (c): GNX - zrckpn).** With the slot empty (N-trained, read empty), GNX is above zrckpn in-domain on MuSiQue (+1.8) and 2Wiki
  (+0.8), level on HotpotQA (both regimes), SQuAD, MetaQA and WebQSP, and below on held-out MuSiQue (-7.0). At the diagonal cells GNX is above in-domain on MuSiQue (R +1.7,
  S +2.1), 2Wiki (+0.8 / +0.9) and HotpotQA (+0.8 / +0.5), level on SQuAD, and on WebQSP level with R and above with S (+1.7). The change of the gap from the empty-slot cell
  to the diagonal cell is level in all 10 in-domain comparisons and in both held-out HotpotQA ones: in-domain, training with the slot does not change GNX's margin over
  zrckpn. It is above on held-out MuSiQue (R +9.4, S +7.3, carried by the N-trained GNX's -7.0) and on MetaQA with R (+8.5), and below on MetaQA with S (-2.2).
- **Beside, no claim (5f's (d): the N-trained pair read with a slot it never trained with).** Above its empty read on 2Wiki (+11.9 to +12.6), HotpotQA (+1.5 to +4.5),
  held-out MuSiQue (GNX +10.0 with either slot; zrckpn +2.9 with R, +2.6 with S) and WebQSP (+2.1 to +5.1; zrckpn with S level); level on in-domain MuSiQue and on SQuAD
  (except zrckpn with S, -0.1 below); on MetaQA GNX +1.0 with R and +4.2 with S, zrckpn level with R and +2.8 with S. At a fixed read slot, the fit trained with that slot
  is above the N-trained fit in 24 of 32 cells, level in 6 and below in 2 (WebQSP with R, -20.8 / -18.2: that family trained with S there). By amendment A2 the N-trained
  fits have no input weights for the slot's two blocks, so these reads feed the links in only through the other parts.
- **Not shown.** A seed spread: seed 0, one fit per model and split, intervals over the questions, so one fit can drive a row (held-out MuSiQue reads, per model, only the fit
  whose training split leaves MuSiQue out). "Empty" removes the structural slot only: N and K stay, and GNX still passes messages over them. 14 of the report's 216 calls
  have an interval end within 0.2 points of 0; in the table these are 2Wiki GNX's training effect (+0.6, from +0.04), SQuAD GNX's three calls and MetaQA zrckpn's R step
  (+0.2, from +0.08).

## 6. What the paper's wording must carry (for the paper session; this stage edits nothing under paper/)

- Name the direction of every swap (trained on which family, read with which).
- "Links matter more than message passing" and "the rule recovers X%" need the b1t baseline named as a read-time shift and need the
  MuSiQue case stated as rule against rule.
- The claim the experiment supports, once Parts A and B are filed: how much of the dataset's own structure a rule-built link recovers for a
  model, and whether message passing adds beyond the same inputs. Nothing about "needing" message passing before the gap-change row is read.

## 7. Disclosure

Development reads on HippoRAG 2's 1,000-question settings (B1 carves), no test split, no LLM. Seeds 0 to 2 of the fits of record on both WQ
splits. The b1t, b1w rows of GNX come from IDPA's reads, unchanged.
