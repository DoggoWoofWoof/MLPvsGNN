# AP2: the models of record read with the rule-built NER slot in the place of spaCy's (b1n, b1x)

Declared 12 November 2026, before any AP2 read. Follows AP1 ([AP1_RULE_NAMES_AND_MUSIQUE_REACH.md](AP1_RULE_NAMES_AND_MUSIQUE_REACH.md), filed in
981dab4) and AP1b ([AP1B_SUBSPAN_NAMES.md](AP1B_SUBSPAN_NAMES.md), filed in 3bb8c25; D76 and D77), and B1L's design
([B1L_MODELS_OF_RECORD_ON_THE_LLM_GRAPH.md](B1L_MODELS_OF_RECORD_ON_THE_LLM_GRAPH.md): `zb1h.py` rebound by import, B1G's `zb1g.py` as the model for a variant
carve). Code: `outputs/mp_unified/zap2.py` (new). Outputs only under `outputs/b1n/`, `outputs/b1x/` and `outputs/ap2/`.

Nothing trains. ML1's and LTM's fits, PKO's chain code, B1P's and B1M's reads of b1w and b1t are read and never written. AP2 decides nothing about
a model, a pool or a column of record. It measures what the models of record get when the NER slot holds a rule-built family.

## 1. Why

The after-paper program (the user, 11 November) starts with "removing spaCy". AP1 and AP1b showed at the graph level that the rule's names, F0T
(U2r's capitalized-run names at the df cap [2, 25], plus each passage's own title as a further name of the passage), link as many of the gold
pairs as spaCy's co-mention family does, and more on MuSiQue and HotpotQA (section 9 of AP1b). A graph-level reach is not a model's recall. The
models of record, zg6r (ML1's GNN) and zrckpn (LTM's MLP), were fitted with spaCy's `ner` store in the NER slot of their training carves. The
question, on HippoRAG 2's three 1,000-question settings (B1): **what do they get when the NER slot of b1w (R + N + K) holds F0T instead of N, with no
new fit?**

## 2. What is read

| carve | structural (R) | NER slot | K | settings | role |
| --- | --- | --- | --- | --- | --- |
| b1t | none | spaCy's N | kNN | musique, 2wiki, hotpotqa | B1P's and B1M's reads; reference |
| b1w | W2_S | spaCy's N | kNN | the same | B1P's and B1M's reads; the base |
| **b1n** | W2_S | **F0T** | kNN | the same | new: Part A |
| **b1x** | W2_S | **F3T** | kNN | musique | new: Part B (D77's default) |

- **The folder.** b1w's six files (`outputs/bench/hipporag2_w/<setting>/`) byte for byte except `graph_ner.npz`, which holds the form's pairs (`src` < `dst`,
  int32, ascending by (src, dst), as b1w's NER file is) and their `weight` (float32). `build.json` is b1w's with the NER file's sha replaced, its `u1a` block
  relabelled (the carve, the variant, this script's sha256; `b1a_structural_sha256` kept as b1w's block holds it) and an `ap2` block (the form and its level tables,
  the pair counts, the AP1b report's sha256 and its `levels.npz` sha256, the sha of b1w's NER file that B1a's lists name).
- **The forms** are AP1b's (`outputs/ap1b/<setting>/levels.npz`; AP1b's report holds each form's pair count). F0T = F0 (U2r's CN_all names) united with
  the level-0 pairs of the names with each passage's own title as a name; F3T = the deepest form (end-anchored and infix sub-spans of two or more
  words, one-word sub-spans, and the titles). A pair's weight is the co-mention weight, the sum of 1/df over the names the two passages share (df
  in [2, 25]), as spaCy's file carries (its weights run from 0.04 = 1/25 up, as the levels' do); where several of a form's level tables hold a pair its weight is
  the deepest table's (at a level, the table with the title before the one without).
- **The models.** zg6r and zrckpn, seeds 0 to 2, fitted on each of the two training splits (WQ-musique, WQ-hotpotqa): 12 reads per setting and carve,
  as B1L's. In-domain = the mean over the fits whose training split is the setting's partner (`zb1h.IN_DOMAIN`: musique reads the WQ-hotpotqa fits, hotpotqa
  the WQ-musique fits, 2wiki both); held-out = the others. The questions are the setting's 1,000.
- **The lists.** B1a's dense and SPLADE lists do not depend on any graph family; `look_b1u.load_lists` checks the setting's other files against the shas
  they name and the replaced structural file against `u1a.b1a_structural_sha256`. AP2's loader (`zap2.lists_of`, its check `zap2.check_lists`) does the same and
  pins the NER file the same way: the lists name the NER file that b1w holds (B1a's own), recorded in `ap2.b1a_ner_sha256`, so a read cannot run on lists built
  for other files. The host's mirror has no SPLADE document vectors, so every host step reads B1a's lists through this loader.

## 3. Gates (a failed gate ends the stage; no read is filed past one)

G1: `levels.npz` loads for the setting, and the built pair count of each form equals AP1b's report (F0T: 98,845 MuSiQue, 42,247 2Wiki, 86,518 HotpotQA;
F3T: 231,258 MuSiQue). G2: the five files other than the NER file equal b1w's `build.json` shas, and b1w's NER file equals B1a's (the lists name it). G2b: the
folder as written carries its own `build.json`'s shas, and B1a's lists check accepts it (the structural and the NER file as the two it replaced). G3: the
NER file has `src` < `dst`, ascending distinct pairs, finite weights at least 0.04 and no pair outside the setting's passages. G4: the node count equals
`build.json`'s. G5: PKO's gate on the carve (`zap2.py gate`, on the host) passes on every setting. G6: the rows recomputed from B1P's and B1M's reads equal
B1M's filed rows, as B1L's check. The self-test passes before any row runs.

## 4. Calls, and what the stage decides

All differences are paired by question, 95% bootstrap intervals over the 1,000 questions (2,000 resamples, seed 0: `b1b.boot_ci`), no threshold.

- **A model's word** on b1n - b1w (in-domain): **GREATER** when the bottom of the interval is above 0, **LESS** when the top is below 0, **SAME** otherwise.
  SAME means only that the interval holds 0; it is not an equivalence.
- **A setting's call**: GREATER, SAME or LESS where the two models agree; **NOT_LESS** where they differ and neither is LESS; **MIXED** where one is LESS
  and the other is not; INCOMPLETE where a read is missing (B1L's `stage_call`).
- **The verdict, Part A: REPLACE** when no setting's call is LESS or MIXED (six in-domain reads, none below), **HOLD** when a setting's call is LESS or MIXED
  (decided even while another setting's read is missing), **INCOMPLETE** while a read is missing and none is LESS or MIXED. REPLACE says the rule-built
  NER slot may stand in the place of spaCy's in the read path of the after-paper program (AP3 builds on it); HOLD keeps spaCy there and names the
  setting and model that fell. This is the WWW rule's shape ("no LOSS among the six deciding reads"), written for paired reads.
- **Reported, not deciding:** the held-out reads; b1n - b1t and b1w - b1t; MuSiQue by hop depth (2, 3, 4 hops; the setting where chain slots matter);
  the gap GNN - MLP at b1n, as B1L reports it beside.
- **Part B (b1x), descriptive:** b1x - b1w and b1x - b1n on MuSiQue, the same words, **no verdict**: S1 refused F3T's price (AP1b section 9) and
  AP2 does not reinterpret it; the read shows what the extra reach (F3T links 1,272 of 1,648 MuSiQue chain slots, F0T 1,173) is worth to a model.

## 5. What it can and cannot show

- It is a **read-time shift**: the models were fitted on spaCy's NER store, so a LESS or SAME here does not say F0T is a worse or an equal training input.
  A refit on carves whose NER slot is rule-built (training carves for all five training datasets rebuilt without spaCy, then the fits) is the user's call (D78;
  default: not now).
- The intervals are over questions, not over seeds: the seeds enter as the mean over fits, as B1L's.
- **Dev-read disclosure.** AP1b chose its forms after a scratch read of MuSiQue's chain labels (D75), so the MuSiQue reads are a dev read of the form. 2Wiki and
  HotpotQA were not looked at for the design; they are the held-out check of the rule.
- Nothing here changes a number of the paper or of any record: b1w's rows remain the record.

## 6. Rows

One block per carve, B1L's rows mirrored (`zb1l`'s names with `b1n` / `b1x`): selftest, settings, fork, chain (per setting, 2 cpus), gate, lists, columns (both
splits), 12 reads per setting (GPU reads 1.1 cpus, GPU share 0.15; MLP reads 6 cpus). Memory is reserved at the measured peak of b1w's same step (B1P's and
B1M's records) scaled by the union's size, B1L's rule (b1n's union holds 1.12 to 1.19 times b1w's pairs, b1x's 2.18 times); the host's measured peak replaces it in
the record. Part B's rows queue after Part A's reads. The report (`zap2.py report`, on the laptop) after the reads.

## 7. Decisions (the user's; my default first)

- **D77** (opened in AP1b section 9): Part B reads F3T on MuSiQue as a disclosed second carve. Default yes, as declared here; the alternative is Part A alone.
- **D78:** a refit with a rule-built NER slot is not part of AP2. Default: not now; after AP2's reads and the user's say-so.

## 8. Development log (disclosed before any read number)

- `zap2.py selftest` (laptop, 12 November, `--variant n` and `--variant x`): the form's pair builder (the union of the level tables, the deepest table's weight,
  `a < b`), the lists check (accepts the replaced structural and NER shas, refuses any other difference), the carve's `build.json`, the settings record's label, the
  paired differences, words, setting calls and the verdict on synthetic reads, and the report's markdown. Passes.
- `zap2.py bench` (laptop, 12 November 01:04 IST, and again at 01:10 after the last code change, so that each folder's recorded script sha is the committed
  file's; the NER files' shas are the same in both runs): the four folders (b1n on MuSiQue, 2Wiki and HotpotQA; b1x on MuSiQue). G1 to G4 and G2b passed. No model, no fit.
- The lists loader (`lists_of`) was run on the four folders against B1a's real lists record (a stand-in for the look modules, its own path only): it loads B1a's
  lists for each and refuses a changed NER sha.
- `zap2.py report` was run once on a temporary fixture (b1n's and b1x's reads made as copies of b1w's reads, so every difference is 0; stub settings records): the
  plumbing, the 92 rows equal to B1M's filed rows, and the report's files. Nothing there is a result; the fixture folders were removed.
- No laptop pass of `settings`, pools or look (B1L's practice: they run on the host; the laptop keeps 24 GB of disk free). `selftest` and `settings` are the block's
  first host items and the chain waits for `settings`. A failing item is fixed, and the fix is disclosed here with its commit, before any read number.

## 9. Part A results (filed 12 November; `outputs/ap2/report.json`, `report.md`, `calls.json`; `zap2.py` as committed in c4dc27c)

**The run.** All 56 Part A rows finished rc 0 on the host (per setting: the chain, PKO's gate, the lists, both column sets and 12 reads, zg6r and zrckpn
with the six fits each); no gate stopped it (G1 to G5 passed before any read; G6: the 92 rows recomputed from B1P's and B1M's reads equal B1M's filed rows). The
report ran on the laptop at 02:32 IST on 12 November on the fetched read files (36 of 36; `Missing reads: 0`); its script sha256 (`8ecb6d86...`) equals the
committed `zap2.py`'s, and the three `build.json` files on the host equal the laptop's byte for byte. Everything below is in `report.md` (R@5 x100, 1,000
questions per setting, 95% bootstrap intervals over the questions). It is a read-time shift (section 5): the models were fitted on spaCy's NER store.

**In-domain** (the mean over the fits that trained on the setting's partner split; b1n - b1w in points with `b1n - b1w`'s interval from `b1b.boot_ci`):

| setting | model | b1t | b1w | b1n | b1n - b1w [95%] | word |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 2wiki | zg6r | 78.1 (76.7-79.5) | 94.9 (94.1-95.7) | 94.7 (93.9-95.5) | -0.25 [-0.51, -0.00] | SAME (a boundary call, below) |
| 2wiki | zrckpn | 77.0 (75.6-78.4) | 94.1 (93.2-94.9) | 94.0 (93.2-94.8) | -0.08 [-0.36, +0.18] | SAME |
| hotpotqa | zg6r | 91.1 (90.0-92.3) | 94.3 (93.3-95.3) | 94.3 (93.3-95.4) | +0.03 [-0.20, +0.28] | SAME |
| hotpotqa | zrckpn | 90.4 (89.2-91.6) | 93.4 (92.4-94.5) | 93.5 (92.4-94.5) | +0.05 [-0.13, +0.23] | SAME |
| musique | zg6r | 66.0 (64.3-67.7) | 69.0 (67.2-70.7) | 68.7 (67.0-70.4) | -0.26 [-0.74, +0.23] | SAME |
| musique | zrckpn | 64.5 (62.8-66.1) | 67.4 (65.7-69.2) | 67.2 (65.5-68.9) | -0.24 [-0.67, +0.18] | SAME |

Against b1t both carves are UP in all six cells by nearly the same amount (b1n - b1t, zg6r and zrckpn: 2Wiki +16.6 and +17.0, HotpotQA +3.2 and +3.1, MuSiQue +2.7
and +2.7; b1w - b1t: +16.8 and +17.1, +3.1 and +3.0, +3.0 and +2.9; every interval is above 0). Each setting's call is SAME (both models SAME), so
**the verdict, Part A, is REPLACE**: no setting's call is LESS or MIXED (section 4). SAME means only that the interval holds 0, not an equivalence: at read time the rule-built slot
gives up between 0.08 and 0.26 points on 2Wiki and MuSiQue in all four cells and nothing on HotpotQA (+0.03 and +0.05).

**The boundary (post hoc: `outputs/ap2/boundary_check.py` and `boundary_check.json`, run after the report; not part of the declared rule).** One word is a boundary
call, zg6r on 2Wiki (-0.25 [-0.51, -0.00]). Its interval's top end is -0.00004 before `b1b.boot_ci` rounds it to four decimals and -0.0 after, and `-0.0 < 0` is false
in the pinned code, so the filed word is SAME. The same resampling (2,000 resamples) for bootstrap seeds 0 to 19 gives LESS in 14 of the 20 at the filed rounding (SAME in 6,
seed 0 among them) and in 20 of 20 before the rounding. The other five cells are SAME in all 20 seeds, rounded or not, with top ends from +0.18 to +0.28 points. Read
the unrounded way, 2Wiki's call would be MIXED (zg6r LESS, zrckpn SAME) and the verdict HOLD. The filed verdict is the pinned code's, REPLACE; the reading to carry is that the
rule-built slot costs the GNN about a quarter of a point on 2Wiki at read time and nothing measurable on the other five cells.

**Reported, not deciding.** Held-out (b1n - b1w, points): HotpotQA zg6r -0.0 [-0.3, +0.3] and zrckpn -0.2 [-0.5, +0.1], MuSiQue zg6r +0.3 [-0.0, +0.7] (all SAME) and
MuSiQue zrckpn +0.4 [+0.0, +0.7] (GREATER). zg6r - zrckpn at b1n: the GNN is above in-domain on all three settings (+0.7, +0.9, +1.5) and held-out on HotpotQA (+0.8); the MLP is
above held-out on MuSiQue (-1.1 [-2.0, -0.1]). MuSiQue by hop depth (in-domain, 518, 316 and 166 questions for 2, 3 and 4 hops): b1n - b1w lies between -0.8 and +0.5, and the
only interval that excludes 0 is the MLP's on 3-hop questions (-0.8 [-1.5, -0.1]). The NER files: F0T holds 42,247 pairs on 2Wiki (spaCy's family in b1w: 36,147), 86,518 on
HotpotQA (68,698) and 98,845 on MuSiQue (75,002), with weight maxima 8.29, 7.68 and 12.15.

**What it decides.** REPLACE says the rule-built NER slot (F0T) may stand in the place of spaCy's in the read path of the after-paper program; AP3 builds on b1n. It
decides nothing about a model, a pool, a column of record or the paper: b1w's rows remain the record. The dev-read disclosure of section 5 stands (MuSiQue is a dev read
of the form; 2Wiki and HotpotQA are its held-out check, both SAME). **D78 stays open** (a refit with a rule-built NER slot in the training carves: the fit would see F0T, which
a read-time shift cannot show). **Part B** (b1x, MuSiQue, descriptive, no verdict) is still running (its chain started at 02:08 IST); it is filed as
section 10 from a second run of the same report script on the same reads, so Part A's rows do not change.

**Files.** `outputs/ap2/report.json`, `report.md` and `calls.json` (written by `zap2.py report`), `boundary_check.py` and `boundary_check.json` (post hoc). The read files stay
on the host under `outputs/b1n/fits/h/` (36 fetched to the laptop, not committed).

## 10. Part B results (filed 12 November; `outputs/ap2/report.json`, `report.md` and `calls.json`, rewritten by a second run of `zap2.py report`; `zap2.py` as committed in c4dc27c)

**The run.** All 20 Part B rows (b1x on MuSiQue: the chain, PKO's gate, the lists, both column sets and 12 reads, zg6r and zrckpn with the six fits each) finished rc 0
on the host by 03:09 IST on 12 November; no gate stopped it. The second report run (laptop, 03:26 IST) read the 12 fetched b1x read files with Part A's 36
(`Missing reads: 0`); its script sha256 equals the committed `zap2.py`'s (`8ecb6d86...`), and b1x's `build.json` on the host equals the laptop's byte for byte
(`2833b030...`). Part A is unchanged: compared field by field with the first run's `report.json`, only `partB` and the time stamp differ (the verdict is still REPLACE;
G6 is still the 92 rows equal to B1M's filed rows). Part B is descriptive, with no verdict (D77's default): b1x - b1w, b1x - b1n and b1x - b1t in points with their
paired 95% intervals and the same words as Part A. It is a read-time shift (section 5).

**MuSiQue, R@5 x100** (b1x = R + F3T + K; F3T holds 231,258 pairs, F0T 98,845 and spaCy's family in b1w 75,002; weight maximum 194.4):

| read | model | b1w | b1n | b1x | b1x - b1w [95%] | b1x - b1n [95%] | b1x - b1t [95%] | words (b1x - b1w, b1x - b1n, b1x - b1t) |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| in-domain | zg6r | 69.0 (67.2-70.7) | 68.7 (67.0-70.4) | 67.2 (65.5-69.0) | -1.7 [-2.3, -1.1] | -1.5 [-2.0, -1.0] | +1.3 [+0.1, +2.5] | LESS, LESS, UP |
| in-domain | zrckpn | 67.4 (65.7-69.2) | 67.2 (65.5-68.9) | 66.8 (65.1-68.5) | -0.7 [-1.2, -0.2] | -0.4 [-0.9, +0.0] | +2.3 [+1.1, +3.4] | LESS, SAME, UP |
| held-out | zg6r | 60.8 (59.1-62.5) | 61.1 (59.4-62.8) | 61.2 (59.5-62.9) | +0.4 [-0.0, +1.0] | +0.1 [-0.3, +0.5] | -0.8 [-2.2, +0.5] | SAME, SAME, FLAT |
| held-out | zrckpn | 61.8 (60.1-63.6) | 62.2 (60.5-64.0) | 62.9 (61.2-64.6) | +1.0 [+0.6, +1.5] | +0.7 [+0.3, +1.1] | +2.8 [+1.6, +4.1] | GREATER, GREATER, UP |

zg6r - zrckpn at b1x, paired by question: in-domain +0.5 [-0.3, +1.3] (LEVEL; at b1w +1.5 [+0.7, +2.4], GNN_ABOVE) and held-out -1.6 [-2.5, -0.8] (MLP_ABOVE; at b1w
-1.1 [-1.9, -0.1]).

**What it says.** In-domain the deepest form gives up read-time R@5 on MuSiQue: LESS than b1w for both models (-1.7 and -0.7 points) and LESS than b1n for the GNN (-1.5),
SAME for the MLP (-0.4, its interval's top end at +0.0); both stay UP against b1t (+1.3 and +2.3). Held-out the MLP is GREATER than b1w and b1n (+1.0 and +0.7) and the GNN
SAME. This is consistent with AP1b's price screen, which refused the sub-span forms (and with them F3T) and carried F0T. It is one setting, a read-time shift, and a
descriptive read: it shows nothing about F3T as a training input (only a refit, D78, would), it sets no setting call, and it changes nothing in Part A's verdict, which
rests on F0T. **D77 is closed by this section:** the second carve was read and F3T is not carried to AP3, whose carves use F0T.

**Files.** The same three report files as section 9 (now with `partB`); the 12 b1x read files stay on the host under `outputs/b1x/fits/h/` (fetched to the laptop, not committed).
