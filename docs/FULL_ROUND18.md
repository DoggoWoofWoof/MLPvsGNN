# Full run: zrc trained on its own entries, only if its screen's re-call is PROMISING and zrm is the base

Declared 8 October 2026 at about 10:00, before any number of round eighteen exists (docs/SCREENS.md, eighteenth
round). The diagnosis zrcd met its declared rule at 09:45: zrm's screen fits read with zrc's entries re-call PROMISING,
and every read but webqsp's is zrm's bit for bit. Round seventeen could not fork zrm's fits as zrc's, so zrc trains.
This run decides, if the screen passes and zrm is the base, whether zrc's gain over zrm holds on every split, and
whether anything else loses. Numbers here are development numbers; the paper's numbers come from one declared
confirmation run.

## 1. Question

zrc is zrm with rmatch's chain entries kept, per row and seed bucket, by w0(c)·m_c(v) in place of m_c(v)
(docs/FULL_ROUND17.md, section 1). Trained on its own entries and read with them: does it lift zrm's reads on every split,
and does nothing else lose? The splits are J5 (in-domain on all five training datasets) and each leave-one-out fit on its
held-out dataset.

## 2. Fits

- **What trains.** zrm's model, settings and training (rmatch.py's train), over zrc's builds (`outputs/zrc/cache`):
  the arm zrc, (zrm.ZRM, zrc.ChainCarveZRC), through `outputs/mp_unified/zrct.py train`. Seed 0, as zrm's fits.
- **The screen's fits are this run's L-musique and L-hotpotqa fits** (`outputs/screen/fits/scr-zrct` and
  `scr-zrct-hp`).
- The run trains four fits, in the order L-2wiki, L-squad, J5, L-metaqa: `zrct.py train --split S --name S --arm zrc
  --out-root outputs/full_zrct/fits`.
- **Only metaqa's training carve differs from zrm's,** in 2 of its 45,883,297 entries. The weights may still differ
  slightly from zrm's, since the training batches differ on two questions.
- **L-metaqa's fit trains no typed graph.** Its training is zrm's, and the match's gates stay at zero. So its six reads
  must be zrm's bit for bit.
- **On the card:**
  - each fit 0.26 (share 0.28) and 9 GB, L-metaqa's 0.18 (share 0.20);
  - each read 0.30 (share 0.32) and 6 GB, as zrcd's reads (zrm's J5 read peaked at 3.9 GB).

## 3. Reads and comparison

- Each fit is read on the six s1eval carves with zrc's entries (`zrct.py read --name S --out-root
  outputs/full_zrct/fits`).
- **Each read is compared, question by question, with zrm's fit of the same split,** p@swa, on the same carves. That
  comparison decides. zret's fit and step 1's are reported beside it.
  - zrm's fits: `outputs/full_zrm/fits/S`; for L-musique and L-hotpotqa, zrm's screen fits `scr-zrm` and `scr-zrm-hp`.
  - zret's fits: `outputs/full_zret/fits/S`; for L-musique, zret's screen fit `scr-zret`.
  - The comparison is lean_screen's: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS or
    WITHIN with the floor 0.0075.
  - **Each comparison also files every read's arrays against zrm's** (`compare-S-same.md`). Reads other than webqsp's
    may differ, through the weights. But **every read of L-metaqa's fit must be IDENTICAL.** If one is not, it is a
    bug: the grade is not filed as a result, and the round stops.
  - hit@1 is reported beside R@5. On webqsp, zrcd's R@5 rose while its hit@1 fell.
  - Twenty-four reads come from these fits and twelve from the screen: 36 in all.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot), and each leave-out
  fit on its held-out dataset.

## 4. Verdict

- `zrct.py grade --full-root outputs/full_zrct --reuse
  L-musique=outputs/screen/scr-zrct.json,L-hotpotqa=outputs/screen/scr-zrct-hp.json` files the grade: relz.py's grade,
  decided against zrm's fits, under zrc's name (zrc.py's).
- **The re-grade decides.** `nullx.py regrade` re-grades all 36 reads under the seed null over every split into
  `outputs/full_zrct/grade-nullx.{md,json}`. Each read's base R@5 is zrm's (`--base-compares`: zrm's six fits compared
  with step 1's, `outputs/zrc/base-zrm-S.json`).
- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

**What ADOPT means here.** The run starts only when zrm is the base (section 5). zrc ADOPT then makes zrc the base of
every later screen and run.

**Every fit runs to its end.** A LOSS filed at the floor 0.0075 may turn WITHIN under the null, so nothing is stopped
early.

## 5. Gates and order of work (8 October)

- **`fzk-gate`** (`zrct.py gate`) exits 0 only when both of these hold:
  - the screen's re-call (`outputs/screen/scr-zrct-pair-recall.json`) is PROMISING;
  - zrm is the base: docs/BASE_ZRM_ZRS.md's re-grade (`outputs/zbase/grade-nullx.json`) is ADOPT.

  Otherwise the feeder drops the four fits with their reads, comparisons and grades. If zrs stays the base, zrc (zrm
  with other entries) is not graded here. Grading it against zrs's fits would be declared in a later round, before its
  numbers.
- **The fits** wait on the gate. Each read waits for its fit. Each comparison waits for its read.
- **The items** go at the end of the feeder's list, after zgs's and zsep's full runs. Nothing is preempted.
- **ETAs if it passes:** the screen's re-call lands about 11:00 to 11:30. Each fit takes about 35 minutes (J5's
  longer) once the card has room. The grade lands about 13:00 to 15:00, depending on the card.

## 6. What this does not do

- No new carves, seeds, variants, columns or hyperparameters of the fit. zrm's model and settings, and rmatch's walk,
  chain cap and relation tables, are used unchanged. Only which 64 entries a row keeps changes.
- No new graph, text, encoder or model. The encoder and the substrate embeddings stay frozen. w0 uses the frozen
  encoder's question and relation embeddings, already on disk.
- Nothing is written into zrm's, zrs's or zret's fit folders.
- webqsp never trains. Test splits are never read.
- No speed figure rests on entries built ahead of time. Any latency figure for this arm is cold (8216ffe): each
  question from scratch, with the walk, w0, the selection and the forward included. Its latency stage is declared in
  its own file.

## Results

Filed 9 October about 03:40 (`outputs/full_zrct/grade.json`, `grade.md`; re-grade `grade-nullx.json`, `grade-nullx.md`).

- **ADOPT**, and **ADOPT under the seed null over all six splits** (section 4). 0 of 36 reads LOSE.
- **The gain is on webqsp, read zero-shot.** J5 +0.0513 (the one primary read that GAINs; floor 0.0295), L-squad
  +0.0406, L-musique +0.0257, L-hotpotqa +0.0450. L-2wiki's +0.0398 GAINs at 0.0075 and turns WITHIN under its null
  floor 0.0473.
- **Everywhere else zrc reads as zrm.** Every other read lies within 0.006 of zrm's. L-metaqa's six reads are
  +0.0000 (why the L-metaqa fit reads as zrm's is not checked here).
- **What this changes.** zrc (zrm with each row keeping the 64 entries w0 ranks highest) is adopted as a full-run result
  against zrm. Whether the MLP track's base moves from zrm to zrc is not decided here. That needs its own declared
  file, as `docs/BASE_ZRM_ZRS.md` was for zrs, committed before any number it reads. Until then zrm stays the MLP's
  base and every screen keeps deciding against zrm's fits.
