# Full run: zret trained with each gold against the non-gold rows (zsep) on all six splits, only if its screen's re-call is PROMISING

Declared 8 October 2026 at about 08:00, before any number of round sixteen's screen exists (docs/SCREENS.md,
sixteenth round). zret is ADOPTED under the seed null over every split, and it is the base of every later screen and
run unless an arm of rounds thirteen to fifteen is ADOPTED. Every arm so far changed the inputs, the model or the loop;
zsep changes only what a question's loss asks of its golds. This run decides, if the screen passes, whether that
objective on zret's model gains on every split with nothing losing. Numbers here are development numbers; the paper's
numbers come from one declared confirmation run.

## 1. Question

zsep is zret's model trained with each gold row set against its question's non-gold rows only. If its screen's re-call
against zret's fits is PROMISING: does its gain over zret hold on every split, and does nothing else lose? The splits
are J5 (in-domain on all five training datasets) and each leave-one-out fit on its held-out dataset.

## 2. Fits

- **Splits:** step 1's six seed-0 splits, with step 1's fit carves and basis.
- **The screen's fits are this run's L-musique and L-hotpotqa fits** (`outputs/screen/fits/scr-zsep` and
  `scr-zsep-hp`). The run trains four fits, in the order L-2wiki, L-squad, J5, L-metaqa, through
  `outputs/mp_unified/zsep.py train --split S --name S --arm zsep --out-root outputs/full_zsep/fits`.
- **Each fit is zret's fit of its split in every respect but the loss:** zret's model (lean_screen3.ZRet), variant p,
  seed 0, config 2e-3:1e-4:0.1:8:2 (Adam at 2e-3 with weight decay 1e-4, dropout 0.1, eight epochs, SWA over epochs 2
  to 7), hidden 128, lean_gpu's loop, batches of 32 questions and guard. The loss is `zsep.listwise_sep` in place of
  `lean_gpu.listwiseD`.
- **A question with one gold in the pool trains as in zret's fit** (the same loss and gradient, up to rounding). Each
  fit records, per training carve, how many questions have two or more golds (screen.json, `zsep`, `census`).
- **On the card:** zret's full-run caps, 0.26 (share 0.28) for L-2wiki, L-squad and J5, and 0.18 (0.20) for
  L-metaqa. Reads 0.28 (0.30).

## 3. Reads and comparison

- Each fit is read on the six s1eval carves.
- Each read is compared, question by question, with zret's fit of the same split, p@swa, on the same carves. That
  comparison decides. Step 1's fit is reported beside it.
  - zret's fits: `outputs/full_zret/fits/S`. For L-musique it is zret's screen fit, `outputs/screen/fits/scr-zret`.
  - The comparison is lean_screen's: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS
    or WITHIN with the floor 0.0075. FC@5's difference (every gold in the top 5) is reported beside it and does not
    decide.
  - Twenty-four reads come from these fits and twelve from the screen: 36 in all.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot), and each leave-out
  fit on its held-out dataset.

## 4. Verdict

- `zsep.py grade --full-root outputs/full_zsep --reuse
  L-musique=outputs/screen/scr-zsep.json,L-hotpotqa=outputs/screen/scr-zsep-hp.json` files the grade: relz.py's
  grade, decided against zret's fits, under zsep's name.
- **The re-grade decides.** `nullx.py regrade` re-grades all 36 reads under the seed null over every split into
  `outputs/full_zsep/grade-nullx.{md,json}`. Each read's base R@5 is zret's (`--base-compares`, zret's six comparisons
  with step 1's fits: `outputs/screen/scr-zret.json` for L-musique, `outputs/full_zret/compare-S.json` for the rest).
- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

**After ADOPT,** zsep becomes the base of every later screen and run: zret's model trained with this loss. If an arm
of rounds thirteen to fifteen (zgs, zrm, zrs) is ADOPTED too, each re-grade is filed and the next base is declared in a
later round, before its numbers. Otherwise zret stays the base (or the arm adopted in rounds thirteen to fifteen).

**The verdicts of rounds thirteen to fifteen do not gate this run.**

**Every fit runs to its end.** A LOSS filed at the floor 0.0075 may turn WITHIN under the null, so no fit is stopped
early.

## 5. Gates and order of work (8 October)

- **`fzp-gate`** (`screen_gate.py pass` on `outputs/screen/scr-zsep-pair-recall.json`) exits 0 only when the re-call
  is PROMISING. The four fits wait on it. On any other result the feeder drops them with their reads, comparisons and
  grades.
- **The items** go at the end of the feeder's list, after zgs's L-hotpotqa fit (docs/SCREENS.md, sixteenth round,
  'Caps and order').
- **ETAs if it passes:** the re-call lands about 09:30 to 10:30. The four fits take about 15 to 25 minutes each and
  their reads about 10 minutes each, beside the other items on the card. The grade comes about 11:00 to 12:00.

## 6. What this does not do

- No new carves, seeds, variants, columns or hyperparameters. zret's model, loop, batches and settings are used as
  they are; only the loss changes.
- Nothing is written into zret's fit folders.
- No new graph, text, encoder or model. The encoder and the substrate embeddings stay frozen.
- webqsp never trains. Test splits are never read.
- No speed figure. zsep serves as zret's model does; any latency figure for it is cold, and its latency stage is
  declared in its own file.

## Results
