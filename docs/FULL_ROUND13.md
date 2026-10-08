# Full run: zret's model trained by gsurg's loop (zgs) on all six splits, only if its screen's re-call is PROMISING

Declared 8 October 2026 at about 06:45, before any number of round thirteen's screen exists (docs/SCREENS.md,
thirteenth round). zret and gsurg are both ADOPTED under the seed null over every split, and they change different
things. So a screen of the two together decides the next base, and this run decides it if the screen passes.
Numbers here are development numbers; the paper's numbers come from one declared confirmation run.

## 1. Question

zgs is zret's model trained by gsurg's loop. If its screen's re-call against zret's fits is PROMISING: does its gain
over zret hold on every split, and does nothing else lose? The splits are J5 (in-domain on all five training
datasets) and each leave-one-out fit on its held-out dataset.

## 2. Fits

- **Splits:** step 1's six seed-0 splits, with step 1's fit carves and basis.
- **The screen's fits are this run's L-musique and L-hotpotqa fits** (`outputs/screen/fits/scr-zgs` and `scr-zgs-hp`).
  The run trains four fits, in the order L-2wiki, L-squad, J5, L-metaqa, through `outputs/mp_unified/zgs.py train
  --split S --name S --arm zgs --out-root outputs/full_zgs/fits`.
- **One fit per split:** variant p, seed 0, config 2e-3:1e-4:0.1:8:2, hidden 128, the SWA state p@swa. No select
  carves, no pick.
- **On the card:** gsurg's full-run caps, 0.26 (share 0.28) for L-2wiki, L-squad and J5, and 0.18 (0.20) for L-metaqa.
  Reads 0.28 (0.30).

## 3. Reads and comparison

- Each fit is read on the six s1eval carves.
- Each read is compared, question by question, with zret's fit of the same split, p@swa, on the same carves. That
  comparison decides. gsurg's fit and step 1's are reported beside it.
  - zret's fits: `outputs/full_zret/fits/S`. For L-musique it is zret's screen fit, `outputs/screen/fits/scr-zret`.
  - gsurg's fits: `outputs/full_gsurg/fits/S`, and its screen fits `scr-gsurg` and `scr-gsurg-hp`.
  - The comparison is lean_screen's: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS
    or WITHIN with the floor 0.0075.
  - Twenty-four reads come from these fits and twelve from the screen: 36 in all.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot), and each leave-out
  fit on its held-out dataset.

## 4. Verdict

- `zgs.py grade --full-root outputs/full_zgs --reuse
  L-musique=outputs/screen/scr-zgs.json,L-hotpotqa=outputs/screen/scr-zgs-hp.json` files the grade: relz.py's grade,
  decided against zret's fits, under zgs's name.
- **The re-grade decides.** `nullx.py regrade` re-grades all 36 reads under the seed null over every split into
  `outputs/full_zgs/grade-nullx.{md,json}`. Each read's base R@5 is zret's (`--base-compares`, zret's six comparisons
  with step 1's fits: `outputs/screen/scr-zret.json` for L-musique, `outputs/full_zret/compare-S.json` for the rest).
- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

**After ADOPT,** zgs becomes the base of every later screen and run. Otherwise zret stays the base.

**Every fit runs to its end.** A LOSS filed at the floor 0.0075 may turn WITHIN under the null, so no fit is stopped
early.

## 5. Gates and order of work (8 October)

- **`fzg-gate`** (`screen_gate.py pass` on `outputs/screen/scr-zgs-pair-recall.json`) exits 0 only when the re-call is
  PROMISING. The four fits wait on it. On any other result the feeder drops them with their reads, comparisons and
  grades.
- **The items** go at the end of the feeder's list, after round twelve's.
- **ETAs if it passes:** the re-call lands about 09:00 to 10:00. The four fits run together and take about 2 to 3.5
  hours, as gsurg's did. The grade comes about 13:00 to 14:30.

## 6. What this does not do

- No new carves, seeds, variants, columns or hyperparameters. zret's model and gsurg's loop are used unchanged.
- No new graph, text, encoder or model. The encoder and the substrate embeddings stay frozen.
- webqsp never trains. Test splits are never read.

## Amendment (8 October about 10:15, before this round's re-call)

zrm became the base at 10:05 (docs/BASE_ZRM_ZRS.md), so a full run against zret's fits can no longer change the base.
This round's re-call is filed as declared. If it is PROMISING, the idea is combined with zrm in a later round, declared
before its numbers, instead of this full run: two screen fits against zrm's, then a full run against zrm's fits. The
full-run items are commented out in the feeder; the gate stays and starts nothing (docs/SCREENS.md, 'Amendment: with
zrm the base').

## Results
