# Full run: rmatch's match on zret's model (zrm) on all six splits, only if its screen's re-call is PROMISING

Declared 8 October 2026 at about 07:05, before any number of round fourteen's screen exists (docs/SCREENS.md,
fourteenth round). zret is ADOPTED under the seed null over every split. rmatch's screen and its re-call are PROMISING
against step 1's fits, and its own full run is under way. This run decides, if the screen passes, whether rmatch's
match holds on zret's base. Numbers here are development numbers; the paper's numbers come from one declared
confirmation run.

## 1. Question

zrm is zret's model with rmatch's question-relation match added. If its screen's re-call against zret's fits is
PROMISING: does its gain over zret hold on every split, and does nothing else lose? The splits are J5 (in-domain on all
five training datasets) and each leave-one-out fit on its held-out dataset.

## 2. Fits

- **Splits:** step 1's six seed-0 splits, with step 1's fit carves and basis.
- **The screen's fits are this run's L-musique and L-hotpotqa fits** (`outputs/screen/fits/scr-zrm` and `scr-zrm-hp`).
  The run trains four fits, in the order L-2wiki, L-squad, J5, L-metaqa, through `outputs/mp_unified/zrm.py train
  --split S --name S --arm zrm --out-root outputs/full_zrm/fits`.
- **One fit per split:** variant p, seed 0, config 2e-3:1e-4:0.1:8:2, hidden 128, the SWA state p@swa. No select
  carves, no pick.
- **L-metaqa's fit trains no typed graph** (metaqa is held out and webqsp never trains). There the match takes no
  gradient and its gates stay at zero, so that fit is zret's model, trained as zret's was.
- **The chains** are rmatch's, as built for the twelfth round. Nothing is rebuilt.
- **On the card:** rmatch's full-run caps, 0.26 (share 0.28) for L-2wiki, L-squad and J5, and 0.18 (0.20) for
  L-metaqa. Reads 0.30 (0.32).

## 3. Reads and comparison

- Each fit is read on the six s1eval carves.
- Each read is compared, question by question, with zret's fit of the same split, p@swa, on the same carves. That
  comparison decides. rmatch's fit and step 1's are reported beside it.
  - zret's fits: `outputs/full_zret/fits/S`. For L-musique it is zret's screen fit, `outputs/screen/fits/scr-zret`.
  - rmatch's fits: `outputs/full_rmatch/fits/S`, and its screen fits `scr-rmatch` and `scr-rmatch-hp`. Each
    comparison waits for rmatch's read of its split.
  - The comparison is lean_screen's: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS
    or WITHIN with the floor 0.0075.
  - Twenty-four reads come from these fits and twelve from the screen: 36 in all.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot), and each leave-out
  fit on its held-out dataset.

## 4. Verdict

- `zrm.py grade --full-root outputs/full_zrm --reuse
  L-musique=outputs/screen/scr-zrm.json,L-hotpotqa=outputs/screen/scr-zrm-hp.json` files the grade: relz.py's grade,
  decided against zret's fits, under zrm's name.
- **The re-grade decides.** `nullx.py regrade` re-grades all 36 reads under the seed null over every split into
  `outputs/full_zrm/grade-nullx.{md,json}`. Each read's base R@5 is zret's (`--base-compares`, zret's six comparisons
  with step 1's fits: `outputs/screen/scr-zret.json` for L-musique, `outputs/full_zret/compare-S.json` for the rest).
- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

**After ADOPT,** zrm becomes the base of every later screen and run. If round thirteen's zgs is ADOPTED too, each
re-grade is filed and the next base is declared in a later round, before its numbers. Otherwise zret stays the base
(or zgs, if it is ADOPTED).

**rmatch's own verdict does not gate this run.** This run makes the same test on zret's base that rmatch's full run
makes on step 1's.

**Every fit runs to its end.** A LOSS filed at the floor 0.0075 may turn WITHIN under the null, so no fit is stopped
early.

## 5. Gates and order of work (8 October)

- **`fzm-gate`** (`screen_gate.py pass` on `outputs/screen/scr-zrm-pair-recall.json`) exits 0 only when the re-call is
  PROMISING. The four fits wait on it. On any other result the feeder drops them with their reads, comparisons and
  grades.
- **The items** go ahead of round thirteen's fits in the feeder's list, behind round twelve's full run
  (docs/SCREENS.md, fourteenth round, 'Caps and order').
- **ETAs if it passes:** the re-call lands about 08:15 to 09:00. The four fits run together beside zgs's and take
  about 30 to 60 minutes. The grade comes about 09:30 to 10:30.
- **Re-queued at about 07:20, before any of its numbers** (docs/SCREENS.md, 'Round fourteen's first smoke failed'):
  the gate is `fzm-gate-b` and the items are `fzm-*-b`, with the same commands and outputs. The re-call now lands about
  08:30 to 09:15. No rule changes.

## 6. What this does not do

- No new carves, seeds, variants, columns or hyperparameters of the fit. zret's model, and rmatch's match and chains,
  are used unchanged. The chains are walks on the existing graph's typed edges, and the relation embeddings are the
  frozen encoder's, already on disk.
- No new graph, text, encoder or model. The encoder and the substrate embeddings stay frozen.
- webqsp never trains. Test splits are never read.
- No speed figure rests on chains built ahead of time. Any latency figure for this arm is cold, as for rmatch
  (docs/FULL_ROUND12.md, section 6). Its latency stage is declared in its own file.

## Results
