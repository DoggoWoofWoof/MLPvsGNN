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

## 5. Part C, the user's call: models trained on S

For the passage datasets the pair of record was never trained on S. The user's direction (trained on S, read with R) needs either refits
of GNX and zrckpn on S + N + K pools (about a day of card time for the six GNX fits alone) or the older SNK-trained pair (zrckp, zspk, 14 Oct;
zrc, zsp, 9 Oct) read with R, which U1a did on HippoRAG 2's three settings only. Until the user chooses, Part A and Part B stand as the
read-time swap on the fits of record, in both directions where they exist (KB: trained on S, R swapped in; passages: trained on R, S swapped in).

## 6. What the paper's wording must carry (for the paper session; this stage edits nothing under paper/)

- Name the direction of every swap (trained on which family, read with which).
- "Links matter more than message passing" and "the rule recovers X%" need the b1t baseline named as a read-time shift and need the
  MuSiQue case stated as rule against rule.
- The claim the experiment supports, once Parts A and B are filed: how much of the dataset's own structure a rule-built link recovers for a
  model, and whether message passing adds beyond the same inputs. Nothing about "needing" message passing before the gap-change row is read.

## 7. Disclosure

Development reads on HippoRAG 2's 1,000-question settings (B1 carves), no test split, no LLM. Seeds 0 to 2 of the fits of record on both WQ
splits. The b1t, b1w rows of GNX come from IDPA's reads, unchanged.
