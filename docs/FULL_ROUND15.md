# Full run: rmatch's match trained on zret's frozen fits (zrs) on all six splits, only if its screen's re-call is PROMISING

Declared 8 October 2026 at about 07:40, before any number of round fifteen's screen exists (docs/SCREENS.md, fifteenth
round). zret is ADOPTED under the seed null over every split. rmatch's full run is NOT_ADOPTED: it GAINs on the typed
graphs in every fit that trains metaqa, and LOSEs on 2wiki read zero-shot in L-2wiki's fit, a graph whose scores the
match itself does not touch. This run decides, if the screen passes, whether the match, trained alone on zret's
finished fits, holds its gains on every split with nothing losing. Numbers here are development numbers; the paper's
numbers come from one declared confirmation run.

## 1. Question

zrs is zret's fit of each split, frozen, with rmatch's question-relation match trained on top of it. If its screen's
re-call against zret's fits is PROMISING: does its gain over zret hold on every split, and does nothing else lose? The
splits are J5 (in-domain on all five training datasets) and each leave-one-out fit on its held-out dataset.

## 2. Fits

- **Splits:** step 1's six seed-0 splits, with step 1's fit carves and basis.
- **The screen's fits are this run's L-musique and L-hotpotqa fits** (`outputs/screen/fits/scr-zrs` and `scr-zrs-hp`).
  The run trains four fits, in the order L-2wiki, L-squad, J5, L-metaqa, through `outputs/mp_unified/zrs.py train
  --split S --name S --arm zrs --out-root outputs/full_zrs/fits`.
- **Each fit's base is zret's fit of its split, p@swa, frozen:** `outputs/full_zret/fits/S`, and for L-musique zret's
  screen fit `outputs/screen/fits/scr-zret`. Training refuses a base trained as another arm, or with other carves,
  basis, blocks, widths, config, seed or hidden size. Every saved state holds the base bit for bit, checked before the
  fit is saved. zret's fit folders are only read.
- **Only the match trains,** with the fit's own settings: variant p, seed 0, config 2e-3:1e-4:0.1:8:2 (Adam at 2e-3
  with weight decay 1e-4 on the match's parameters, eight epochs, SWA over epochs 2 to 7), hidden 128, lean_gpu's
  listwise loss and guard, batches of 32 of the typed training carves' questions (metaqa's; webqsp never trains). The
  frozen model scores as it reads, without dropout. The SWA state p@swa averages the match's parameters and keeps the
  base.
- **L-metaqa's fit trains no typed graph.** The match takes no step and its gates stay at zero, so that fit is zret's
  fit bit for bit, and its six reads equal zret's.
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
- **On squad, musique, hotpotqa and 2wiki every read equals zret's bit for bit, by construction** (section 2). Those
  reads are WITHIN at a difference of exactly 0, and the results say so: they are not evidence that the match
  transfers to graphs without typed relations.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot), and each leave-out
  fit on its held-out dataset.

## 4. Verdict

- `zrs.py grade --full-root outputs/full_zrs --reuse
  L-musique=outputs/screen/scr-zrs.json,L-hotpotqa=outputs/screen/scr-zrs-hp.json` files the grade: relz.py's grade,
  decided against zret's fits, under zrs's name.
- **The re-grade decides.** `nullx.py regrade` re-grades all 36 reads under the seed null over every split into
  `outputs/full_zrs/grade-nullx.{md,json}`. Each read's base R@5 is zret's (`--base-compares`, zret's six comparisons
  with step 1's fits: `outputs/screen/scr-zret.json` for L-musique, `outputs/full_zret/compare-S.json` for the rest).
- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

**After ADOPT,** zrs becomes the base of every later screen and run: zret's fits with the match trained on them. If zgs
or zrm is ADOPTED too, each re-grade is filed and the next base is declared in a later round, before its numbers.
Otherwise zret stays the base (or the arm adopted in round thirteen or fourteen).

**zrm's verdict does not gate this run,** and rmatch's (NOT_ADOPTED) did not gate its declaration.

**Every fit runs to its end.** A LOSS filed at the floor 0.0075 may turn WITHIN under the null, so no fit is stopped
early.

## 5. Gates and order of work (8 October)

- **`fzs-gate`** (`screen_gate.py pass` on `outputs/screen/scr-zrs-pair-recall.json`) exits 0 only when the re-call is
  PROMISING. The four fits wait on it. On any other result the feeder drops them with their reads, comparisons and
  grades.
- **The items** go after round fourteen's and ahead of zgs's L-hotpotqa fit in the feeder's list (docs/SCREENS.md,
  fifteenth round, 'Caps and order'). Each training item lists zret's fit of its split (`models.pt` and `screen.json`)
  among its inputs, so the feeder waits for them and records their sha at launch.
- **ETAs if it passes:** the re-call lands about 08:30 to 09:30. The four fits take minutes each and their reads about
  10 minutes each, beside the other items on the card. The grade comes about 09:30 to 10:30.

## 6. What this does not do

- No new carves, seeds, variants, columns or hyperparameters. zret's fits are used as they are, and rmatch's match and
  chains unchanged. The chains are walks on the existing graph's typed edges, and the relation embeddings are the
  frozen encoder's, already on disk.
- Nothing is written into zret's fit folders.
- No new graph, text, encoder or model. The encoder and the substrate embeddings stay frozen.
- webqsp never trains. Test splits are never read.
- No speed figure rests on chains built ahead of time. Any latency figure for this arm is cold, as for rmatch
  (docs/FULL_ROUND12.md, section 6). Its latency stage is declared in its own file.

## Results

**ADOPT (09:38).** `outputs/full_zrs/grade-nullx.md`; filed ADOPT in `grade.md`.
- Two primary GAINs: J5's metaqa in-domain (+0.1205 R@5) and J5's webqsp read zero-shot (+0.0912). No LOSS among the
  36 reads, and no call changed under the null.
- metaqa in-domain GAINs in all five fits that train it (+0.114 to +0.122), and webqsp read zero-shot GAINs in all five
  (+0.077 to +0.091).
- Every untyped read, and every read of L-metaqa's fit, is zret's bit for bit.
- zrs becomes the base of every later screen and run, unless zrm's re-grade (docs/FULL_ROUND14.md) is ADOPT too. In
  that case the next base is declared in a later round, before its numbers.

docs/SCREENS.md has the details under 'zrs's full run'.
