# A6: the words around a link, as a fixed input to both models

Declared 17 October 2026, before any number. The sixth reopening of docs/VERDICT_AUDIT_2026_10_17.md (section 2,
row 6), under docs/SCREENS.md section 3b. Code: `outputs/mp_unified/za6lw.py`. Filed 18 October: both arms NOT_ADOPTED (section 7).

## 1. Why this, now

D4 (docs/DIAG_LINK_WORDS.md, 13 October) asked whether the words around a link say which neighbour a question
follows. It scored a link between a found row a and a candidate c by the largest summed IDF of the question's stems
in one window of the link, with both ends' title stems left out. Its rule read one number: the share of musique's
missed golds ranked 1st or 2nd among the found golds' neighbours, fused with zrc's score, against a bar of +0.050 and
1.25 times. It filed **NO_SIGNAL**:

| fit | le2, zrc's score | le2, words alone | le2, fused |
| --- | ---: | ---: | ---: |
| scr-zrct | 0.225 | 0.294 | 0.261 (+0.036) |
| scr-zrct-hp | 0.291 | 0.300 | 0.308 (+0.018) |

- **Beside, on the hyperlink graphs,** the fused order was above zrc's score in both fits (2wiki 0.733 and 0.737
  against 0.706 and 0.697; hotpotqa 0.599 and 0.578 against 0.583 and 0.549).
- **On the KB graphs** the relation names led away from the gold (metaqa le2 0.447 → 0.127).
- **The bar was set for a fused score, not a model input.** The words never reached a model; a model can learn the
  weight per family that D4's fixed sum could not. Under section 3b the idea is open until it reaches both models.

## 2. The form

Six columns per row, computed once from the first stage's list, before either model scores a row:

- **The anchors:** the question's top five rows by rrf among the retrieved rows (rrf > 0), ties by row. No label and
  no model score: D4's anchors were the found golds (an oracle) or zrc's top-1; A6's are the first stage's.
- **The link score** w_f(v), per family f (structural, NER): over the anchors a other than v that share a link of
  family f with v in the pool, the largest of D4's link score (`linkwords.link_score`, unchanged), 0 without one.
  - structural: the 10 tokens on each side of each occurrence of one end's title in the other's text, both ways; on
    the typed graphs (metaqa, webqsp), the names of the relations in the edge's slots;
  - NER: the 10 tokens on each side of each mention, in both texts, of an entity both mention (df within [2, 25]);
  - kNN: no words.
- **The columns:** log1p w_s, log1p w_n, w_s / Q, w_n / Q (Q the question's total stem IDF, 0 when it is 0), and a
  flag per family that v has a window with an anchor. Stored as float16.

**The NER mentions, on all six corpora.** D4 had mentions for musique and squad only. A6 runs spaCy 3.8.15 with
en_core_web_sm 3.8.0 as the graph's NER family was built (parser and lemmatizer off, D4's nine labels, at least three
characters, key `e.text.lower().strip()`) on every node of all six corpora:
- in parts (metaqa 1, squad 1, musique 2, webqsp 10, hotpotqa 40, 2wiki 60), one CPU each, on the host;
- merged into D4's format (keys sorted, df the distinct nodes per key, rows in node order), with the nodes' sha256
  checked;
- **kept only when they rebuild at least 0.9 of the graph's NER edges** (D4's REPRO_MIN). Below it the merge writes a
  gate record, no mentions, and exits 1, so nothing downstream runs. On musique and squad the merge also reports
  whether its arrays are D4's, bit for bit.

**Into both models: R1's pointwise head.** Linear(6, 16), GELU, Linear(16, 1): 129 parameters, the last layer zero at
the start, the first drawn from its own generator (seed + 6601), so from the same seed each arm's trunk is its base's
and each arm starts as its base bit for bit.

**The MLP rule.** The columns are the row's own, from the first stage's list and the pool's links, before any row is
scored. No score input; nothing is normalised over the question's scores (Q is the question's, not its rows').
The MLP reads the links' words as fixed numbers; it passes no message.

## 3. The arms

| arm | model | carve |
| --- | --- | --- |
| zrckpw | zrckp's model plus the head at the top (MRO: ZRCKPW, ZRCKP, …) | zrckp's (`zp1c3.KTChainCarveZRC`) with the columns beside |
| zspkw | zspk's model with the head added to the KT head before zsp's propagation reads the score (MRO: ZSPKW, ZProp, A6Mix, KTMix, ZRM, …) | zspk's (`zp1c3.KTLinkCarveBase`) with the columns beside |

zspkw's head sits where R1's did: the propagation spreads it along the pool's links, so the GNN can pass a worded
link's score on.

## 4. The fits and the call

- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-a6-<arm>`, `-hp`), on P1c2's roots with
  P1c3's KT input. Card fit settings: variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128.
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split, which decides:
  - zrckp: scr-p1c3p-zrckp and -hp;
  - zspk: scr-p1c3-zspk and -hp.

  Step 1's fit is reported beside. The re-call's base screens are L1's.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), per arm. GAIN,
  LOSS and WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A6's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- musique, 2wiki and hotpotqa, both splits and arms (D4's reads);
- musique in domain (L-hotpotqa), both arms: Claim 1's read;
- hit@1 and FC@5 per read;
- per build: the shares of rows with a window and with a word, per family; the anchors per question; the float16
  error; per merge: the NER edges rebuilt.

**Serving cost** (reported later, not part of this call): the columns need the pool's anchor links' texts and
mentions at query time; the timed form is declared with the full run if one follows.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools (the columns built again on P1g-b's looks), then its full run |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A6k, the back-off form: the structural columns split by their reading (relation names on the typed graphs, title windows on the others; eight columns), so the KB graphs' words, which D4 found lead away from the gold, cannot move the passage graphs' weights; both models |
| NOT_ADOPTED, with no GAIN | A6e: on zspk, each pool edge's own link score (between its two ends) as a weight in zsp's propagation, a message along the worded link; on zrckp, the anchors widened to the first stage's top ten rows |
| INCOMPLETE | the missing re-call is run; nothing else changes |

- **If both arms pass,** the columns join both lean models.
- **If only one passes,** its full run is declared; the next file decides whether F1's rule needs the other model to
  read the columns too.
- **What does not happen.** No model change is adopted on today's pools (the user's rule of 13 October).

## 6. How it runs

**Self-test.** `python outputs/mp_unified/za6lw.py --selftest` passes before this commit (6 s). It checks:
- the arms' MROs; the base arms unchanged; 129 head parameters;
- the mentions: three parts merged are a one-shot build's arrays (`linkwords.aggregate`) array for array; D4's loader
  reads the file; the merge refuses another part count, a changed part, a node file that is not the manifest's, and
  a graph it does not rebuild (no file kept, a gate record written);
- the columns against a brute force over every (row, anchor, family), on rmatch's typed toy look with pools and an
  rrf column, in the passage reading (title windows) and the KB reading (relation names); the KB reading moves only
  the structural columns; the build reads exactly the anchors and their link ends; one part at a time gives the whole
  build's arrays byte for byte; the anchors are the top five by rrf with ties by row;
- the build's refusals: a question in no train/dev/validation file, a cache part built from another look, a relation
  slot past the vocabulary, a part number past the carve's;
- a batch's columns are its rows' own on both carves;
- each arm starts as its base bit for bit; with the head set, zrckpw adds it and zspkw's propagation reads the score
  with it added; without the columns each is its base;
- toy fits through lean_gpu's loop repeat bit for bit and move the head; without the columns each fit is its base's;
- the refusals; relz's pair and re-call per arm, stamped A6's; the verdict's outcomes and next forms.

**Laptop check, passed before this commit:** squad's mentions, rebuilt in A6's one part and merged, are D4's squad
file (made on the laptop with the same settings) bit for bit: 188,101 mentions, 64,439 keys, every one of the graph's
156,130 NER edges rebuilt and none extra. 315 s on one laptop CPU, 15.6 ms a passage.

**Host items** (177, `a6-*`), after A5's verdict:
- `a6-spacy`: unpacks the pushed spaCy closure once (sha256 checked) and parses one sentence;
- 114 NER parts, 1 CPU and 1.5 GB each: metaqa, squad and musique first (so the chain runs end to end early), then
  2wiki and hotpotqa interleaved, webqsp last; about 45 CPU-hours in all (D4's laptop rate, 12 to 19 ms a
  passage);
- 6 merges (2 to 12 GB); 39 column builds, one per step-1 cache part (R1's part counts), 3 to 10 GB;
- the 17 screen items, the trains gated on their split's fit-carve builds and the reads on the six s1eval carves'
  builds. GPU sizing as A3's to A5's, from the base arms' measured peaks: L-musique trains `--frac 0.20` (share
  0.22), -hp trains and the reads `--frac 0.27` (0.29); memory: trains 4.5 GB (zspkw-hp 7 GB), reads 6 GB.

## 7. Results

Filed 18 October 2026 from the host's records (`outputs/a6/screen-verdict.json`, written 2026-10-17T19:33:02Z; the
re-calls `outputs/screen/scr-a6-zrckpw-pair-recall.json` and `-zspkw-`, host paths made relative). **Both arms
NOT_ADOPTED, with no GAIN.**

- **zrckpw: 0 GAIN, 2 LOSS,** both on webqsp zero-shot: -0.0188 (L-musique) and -0.0264 (L-hotpotqa).
- **zspkw: 0 GAIN, 1 LOSS,** webqsp zero-shot -0.0107 on L-musique (floor 0.0102); -0.0021 on L-hotpotqa.
- **On the passage graphs** every read is WITHIN, most of them up a little: hotpotqa in domain +0.0025 and +0.0031
  (L-musique), hotpotqa zero-shot +0.0024 and +0.0046 (L-hotpotqa). musique moves under 0.004 either way.

So the link words help no read past its floor, and the only losses are on webqsp, whose structural words are the
names of the relations in an edge's slots: the words D4 found lead away from the gold on the typed graphs. This closes
only this form (D4's link score to the first stage's top five rows, six columns through R1's pointwise head). The
next form, fixed in section 5 for no GAIN, is **A6e**: on zspk, each pool edge's own link score as a weight in zsp's
propagation; on zrckp, the anchors widened to the first stage's top ten rows. Because the losses sit where A6k's split
was written for, A6e's file also splits the structural columns by their reading (relation names on the typed graphs,
title windows on the others), so the KB words cannot move the passage graphs' weights. That choice is made here,
before A6e's numbers, and A6e is declared in its own file.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's; base R@5 shown as zrckp's /
zspk's):

| split | dataset | read | base R@5 | zrckpw: delta [95% CI], call | zspkw: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8019 / 0.8009 | +0.0001 [-0.0007, +0.0008], WITHIN | +0.0017 [+0.0009, +0.0026], WITHIN |
| L-musique | squad | in-domain | 0.9116 / 0.9114 | +0.0002 [-0.0018, +0.0021], WITHIN | +0.0012 [-0.0009, +0.0032], WITHIN |
| L-musique | musique | zero-shot | 0.5121 / 0.5012 | -0.0011 [-0.0062, +0.0037], WITHIN | -0.0031 [-0.0088, +0.0025], WITHIN |
| L-musique | hotpotqa | in-domain | 0.9065 / 0.9080 | +0.0025 [+0.0002, +0.0047], WITHIN | +0.0031 [+0.0010, +0.0051], WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 / 0.9383 | +0.0011 [-0.0005, +0.0026], WITHIN | -0.0020 [-0.0034, -0.0007], WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 / 0.3229 | -0.0188 [-0.0298, -0.0078], LOSS | -0.0107 [-0.0189, -0.0024], LOSS |
| L-hotpotqa | metaqa | in-domain | 0.8044 / 0.8040 | -0.0022 [-0.0031, -0.0015], WITHIN | +0.0003 [-0.0004, +0.0012], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 / 0.9135 | +0.0007 [-0.0013, +0.0026], WITHIN | -0.0008 [-0.0028, +0.0010], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 / 0.5754 | +0.0006 [-0.0042, +0.0054], WITHIN | -0.0006 [-0.0052, +0.0043], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 / 0.8499 | +0.0024 [-0.0006, +0.0053], WITHIN | +0.0046 [+0.0016, +0.0076], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 / 0.9333 | +0.0018 [+0.0004, +0.0033], WITHIN | -0.0058 [-0.0074, -0.0043], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 / 0.2680 | -0.0264 [-0.0379, -0.0151], LOSS | -0.0021 [-0.0113, +0.0064], WITHIN |
