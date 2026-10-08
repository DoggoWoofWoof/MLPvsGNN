# Full run: zrk, an objective for recall at five on zrm, only if its screen's re-call is PROMISING and zrm is the base

Declared 8 October 2026 at about 11:33, before any number of round twenty exists (docs/SCREENS.md, twentieth round).
zrm is the base (docs/BASE_ZRM_ZRS.md, re-graded ADOPT at 10:05). This run decides, if the screen passes, whether zrk's
gain over zrm holds on every split, and whether anything else loses. Numbers here are development numbers; the paper's
numbers come from one declared confirmation run.

## 1. Question

zrk is zrm trained with a smooth recall at five added to its loss. Does training for the metric the screens decide on
lift zrm's reads on every split, and does nothing else lose? The splits are J5 (in-domain on all five training datasets)
and each leave-one-out fit on its held-out dataset.

Why this question:
- **zrm loses most of its R@5 on questions it finds only in part.** That is 0.737 and 0.765 of musique's lost R@5 in
  zrm's screen fits, and 0.875 to 0.887 of 2wiki's (outputs/diag/goldsplit-zrm).
- **lean_gpu's loss trains for R@5 only through a softmax over the whole pool.** Each gold's term pushes on every row
  by its share, so most of the push goes to the rows already on top.
- **The change is general.** Every training question on every dataset takes the same loss, with settings fixed before
  any number.

## 2. Fits

- **What trains.** The arm zrk: zrm's model and carve (zrm.ZRM over rmatch.ChainCarveBase), settings and training
  (rmatch.py's train, over rmatch's builds), seed 0, through `outputs/mp_unified/zrk.py train`.
- **The loss.** listwiseD + 1 x (1 - smooth R@5), averaged over the questions with a gold:
  - a gold's rank in its pool is 1 plus a sigmoid sum, temperature 0.1 score units, over the pool's other rows
    (ApproxNDCG's form, Qin, Liu and Li 2010);
  - "rank at most five" is a sigmoid, temperature 1 rank, of 5.5 minus that rank (the Recall@k surrogate of Patel,
    Tolias and Matas, CVPR 2022);
  - another gold above a gold counts toward its rank but carries no gradient. Two golds that swap never change R@5.
- **A check before training.** Each fit checks the loss and its gradient on the first eight questions of every training
  carve, at the fit's start, against a float64 reference, and stops (exit 1) on a mismatch. Each fit records its
  training carves' census in screen.json.
- **The screen's fits are this run's L-musique and L-hotpotqa fits** (`outputs/screen/fits/scr-zrk` and `scr-zrk-hp`).
- The run trains four fits, in the order L-2wiki, L-squad, J5, L-metaqa: `zrk.py train --split S --name S --arm zrk
  --out-root outputs/full_zrk/fits`.
- **The loss acts in training only.** Reads and serving are zrm's.
- **On the card,** as zrm's full-run fits (torch peaks 3.1 to 5.4 GB):
  - each fit 0.26 (share 0.28) and 9 GB, L-metaqa's 0.18 (share 0.20);
  - each read 0.30 (share 0.32) and 6 GB.

## 3. Reads and comparison

- Each fit is read on the six s1eval carves with rmatch's chain entries, as zrm's fits were (`zrk.py read --name S
  --out-root outputs/full_zrk/fits`).
- **Each read is compared, question by question, with zrm's fit of the same split,** p@swa, on the same carves. That
  comparison decides. zret's fit and step 1's are reported beside it.
  - zrm's fits: `outputs/full_zrm/fits/S`; for L-musique and L-hotpotqa, zrm's screen fits `scr-zrm` and `scr-zrm-hp`.
  - zret's fits: `outputs/full_zret/fits/S`; for L-musique, zret's screen fit `scr-zret`.
  - The comparison is lean_screen's: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS or
    WITHIN with the floor 0.0075.
  - hit@1 is reported beside R@5.
  - Twenty-four reads come from these fits and twelve from the screen: 36 in all.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot), and each leave-out
  fit on its held-out dataset.

## 4. Verdict

- `zrk.py grade --full-root outputs/full_zrk --reuse
  L-musique=outputs/screen/scr-zrk.json,L-hotpotqa=outputs/screen/scr-zrk-hp.json` files the grade: relz.py's grade,
  decided against zrm's fits, under zrk's name (zrc.py's mapping).
- **The re-grade decides.** `nullx.py regrade` re-grades all 36 reads under the seed null over every split into
  `outputs/full_zrk/grade-nullx.{md,json}`. Each read's base R@5 is zrm's (`--base-compares`: zrm's six fits compared
  with step 1's, `outputs/zrc/base-zrm-S.json`).
- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

**What ADOPT means here.**
- zrk ADOPT makes zrk the base of every later screen and run.
- If another arm of these rounds (zrc, zkind) is ADOPTED against zrm as well, the base is
  decided in a file declared before that decision, as docs/BASE_ZRM_ZRS.md decided between zrm and zrs. zrk changes
  only the loss, so it would combine with any of them in a round of its own.
- If another arm becomes the base before this run is graded, this run still decides zrk against zrm's fits, as
  declared. A zrk ADOPT then leads to a combination with that base in a later round (docs/BASE_ZRM_ZRS.md, section 4).

**Every fit runs to its end.** A LOSS filed at the floor 0.0075 may turn WITHIN under the null, so nothing is stopped
early.

## 5. Gates and order of work (8 October)

- **`frk-gate`** (`zrk.py gate`) exits 0 only when both of these hold:
  - the screen's re-call (`outputs/screen/scr-zrk-pair-recall.json`) is PROMISING;
  - zrm is the base: docs/BASE_ZRM_ZRS.md's re-grade (`outputs/zbase/grade-nullx.json`) is ADOPT. It is (10:05).

  Otherwise the feeder drops the four fits with their reads, comparisons and grades.
- **The fits** wait on the gate. Each read waits for its fit. Each comparison waits for its read.
- **The items.** The screen's items go after round nineteen's screen items, ahead of the queued full runs' fits. The
  full run goes at the end of the feeder's list. Nothing is preempted.
- **ETAs.**
  - The screen's two fits start when the card has room, about 11:55 to 12:15, and take about 35 minutes each. The
    re-call lands about 12:45 to 13:15.
  - If it passes, the four fits queue behind the full runs already queued. The grade lands about 15:00 to 18:00,
    depending on the card.

## 6. What this does not do

- No new column, block, carve, graph, text, seed or variant. zrm's model, settings and training, and rmatch's walk,
  chain cap and relation tables, are used unchanged. The loss's four settings (K 5, temperatures 0.1 and 1, weight 1)
  are fixed here and not searched.
- No new graph, text, encoder or model. The encoder and the substrate embeddings stay frozen.
- Nothing is written into zrm's, zrs's or zret's fit folders.
- webqsp never trains. Test splits are never read.
- **Speed.** The loss changes training only. zrk reads and serves as zrm does, so any latency figure for it is zrm's,
  and cold (8216ffe): each question timed from scratch, with the walk, the move of its entries to the device and the
  forward included, no warm-up pass, and nothing kept from an earlier question.

## Results

- **The screen's re-call: NO_GAIN (19:39; docs/SCREENS.md).** No GAIN and no LOSS among the twelve reads; every read
  moves by 0.0069 R@5 or less. `frk-gate` exits 1, and this full run does not run.
