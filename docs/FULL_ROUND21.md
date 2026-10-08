# Full run: zrg, gsurg's loop on zrm, only if its screen's re-call is PROMISING and zrm is the base

Declared 8 October 2026 at about 11:55, before any number of round twenty-one exists (docs/SCREENS.md,
twenty-first round). zgs's screen (round thirteen, gsurg's loop on zret's model) re-called PROMISING at 11:47. By the
amendment of 10:15 (docs/SCREENS.md, 'Amendment: with zrm the base'), a PROMISING re-call of zgs leads to a combination
with zrm, the base (docs/BASE_ZRM_ZRS.md, re-graded ADOPT at 10:05), declared before its numbers. This is that round.
This run decides, if its own screen passes, whether zrg's gain over zrm holds on every split, and whether anything else
loses. Numbers here are development numbers; the paper's numbers come from one declared confirmation run.

## 1. Question

zrg is zrm trained by gsurg's loop. At each step, the gradient on each training dataset's questions loses its component
along the gradient of every other dataset in the step that it conflicts with (lean_screen5.fit_gsurg). gsurg was ADOPTED
on its own (06:21); zgs put it on zret's model, and its screen re-called PROMISING. Does the loop lift zrm's reads on
every split, and does nothing else lose? The splits are J5 (in-domain on all five training datasets) and each
leave-one-out fit on its held-out dataset.

## 2. Fits

- **What trains.** The arm zrg, (zrm.ZRM, rmatch.ChainCarveBase), through `outputs/mp_unified/zrg.py train`: zrm's
  model, settings and training (rmatch.py's train) run inside lean_screen5's gsurg_loop. Seed 0, as zrm's fits.
  - Each epoch's surgery counts land in train.json's curve. A fit whose curve lacks them trained outside the loop and
    stops (exit 1).
- **The screen's fits are this run's L-musique and L-hotpotqa fits** (`outputs/screen/fits/scr-zrg` and `scr-zrg-hp`).
- The run trains four fits, in the order L-2wiki, L-squad, J5, L-metaqa: `zrg.py train --split S --name S --arm zrg
  --out-root outputs/full_zrg/fits`.
- **The loop acts in training only.** Reads and serving are zrm's.
- **On the card:** each fit 0.30 (share 0.32) and 9 GB, L-metaqa's 0.18 (share 0.20) as gsurg's; each read 0.30 (share
  0.32) and 6 GB. zgs's L-hotpotqa fit in the loop reserved 5.5 GB of its 6.2 GB cap (0.26), and zrm's chain entries add
  to the loop's batches.

## 3. Reads and comparison

- Each fit is read on the six s1eval carves with rmatch's chain entries, as zrm's fits were (`zrg.py read --name S
  --out-root outputs/full_zrg/fits`).
- **Each read is compared, question by question, with zrm's fit of the same split,** p@swa, on the same carves. That
  comparison decides. zret's fit and step 1's are reported beside it (zgs's screen fits beside the screen's reads).
  - zrm's fits: `outputs/full_zrm/fits/S`; for L-musique and L-hotpotqa, zrm's screen fits `scr-zrm` and `scr-zrm-hp`.
  - The comparison is lean_screen's: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS or
    WITHIN with the floor 0.0075.
  - hit@1 is reported beside R@5.
  - Twenty-four reads come from these fits and twelve from the screen: 36 in all.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot), and each leave-out
  fit on its held-out dataset.

## 4. Verdict

- `zrg.py grade --full-root outputs/full_zrg --reuse
  L-musique=outputs/screen/scr-zrg.json,L-hotpotqa=outputs/screen/scr-zrg-hp.json` files the grade: relz.py's grade,
  decided against zrm's fits, under zrg's name (zrc.py's mapping).
- **The re-grade decides.** `nullx.py regrade` re-grades all 36 reads under the seed null over every split into
  `outputs/full_zrg/grade-nullx.{md,json}`. Each read's base R@5 is zrm's (`--base-compares`:
  `outputs/zrc/base-zrm-S.json`).
- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

**What ADOPT means here.** zrg ADOPT makes zrg the base of every later screen and run. If another arm of these rounds
(zrc, zkind, zrk) is also ADOPTED against zrm, the base is decided in a file declared before that decision, as
docs/BASE_ZRM_ZRS.md decided between zrm and zrs. If another arm becomes the base before this run is graded, this run
still decides zrg against zrm's fits, as declared, and a zrg ADOPT leads to a combination with that base in a later
round.

**Every fit runs to its end.** A LOSS filed at the floor 0.0075 may turn WITHIN under the null, so nothing is stopped
early.

## 5. Gates and order of work (8 October)

- **`fzrg-gate`** (`zrg.py gate`) exits 0 only when both of these hold:
  - the screen's re-call (`outputs/screen/scr-zrg-pair-recall.json`) is PROMISING;
  - zrm is the base: docs/BASE_ZRM_ZRS.md's re-grade (`outputs/zbase/grade-nullx.json`) is ADOPT. It is (10:05).

  Otherwise the feeder drops the four fits with their reads, comparisons and grades.
- **The fits** wait on the gate. Each read waits for its fit. Each comparison waits for its read.
- **The items.** The screen's items go after round eighteen's full-run items, which are under way, and ahead of the
  later full runs' fits (rounds nineteen and twenty): the screen's fits take about two hours each, so ahead of round
  eighteen's remaining fits they would delay its grade by about an hour. The full run goes at the end of the feeder's
  list. Nothing is preempted.
- **ETAs.** The screen's two fits start once round eighteen's last fits have started, about 12:30 to 13:15, and take
  about 2 to 2.5 hours each (zgs's L-hotpotqa fit took 2.1 hours: 8 epochs of about 15.5 minutes). The re-call lands
  about 15:00 to 16:00. If it passes, the four fits queue at the end of the list, behind the full runs of rounds
  nineteen and twenty if those pass, and the grade lands about 20:00 to 24:00, depending on the card.

## 6. What this does not do

- No new column, block, carve, graph, text, seed, variant or hyperparameter. zrm's model and settings and gsurg's loop
  are used unchanged.
- No new graph, text, encoder or model. The encoder and the substrate embeddings stay frozen.
- Nothing is written into zrm's, zrs's, zret's or zgs's fit folders.
- webqsp never trains. Test splits are never read.
- **Speed.** The loop changes training only: about 3.5 times lean_gpu's time per epoch, since each step takes every
  other present dataset's gradient. zrg reads and serves as zrm does, so any latency figure for it is zrm's, and cold
  (8216ffe): each question timed from scratch, with the walk, the move of its entries to the device and the forward
  included, no warm-up pass, and nothing kept from an earlier question.

## Results
