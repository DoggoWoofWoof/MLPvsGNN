# Full run: the question-to-depth prior (qdepth or relqd) on all six splits, only if its screen's re-call is PROMISING

Declared 8 October 2026 at about 03:50, before any number of round eleven's screen exists (docs/SCREENS.md, eleventh
round). The arm runs on one base, named by rel's re-grade under the seed null over every split
(`outputs/full_rel/grade-nullx`, docs/FULL_ROUND9.md):
- **ADOPT:** relqd, the prior on rel's carve and blocks, decided against rel's fits;
- **NOT_ADOPTED:** qdepth, the prior on step 1's, decided against step 1's fits;
- **missing or INCOMPLETE:** neither.

The run starts only if that base's pair, re-called under the seed null (`outputs/screen/scr-qdepth-pair-recall` or
`outputs/screen/scr-relqd-pair-recall`), is PROMISING. Otherwise none of it happens, and this file records why.
Numbers here are development numbers; the paper's numbers come from one declared confirmation run.

## 1. Question

Each row's score gains a learned depth prior for its question, l_q[depth(v)] with l_q = q U + P_q W + b
(docs/SCREENS.md, eleventh round). If the screen's re-call is PROMISING: does its gain over the base hold on every
split, and does nothing else lose? The splits are J5 (in-domain on all five training datasets) and each leave-one-out
fit on its held-out dataset. The screen reads musique, hotpotqa and webqsp zero-shot. Only this run reads metaqa,
squad and 2wiki zero-shot, and every dataset in-domain under J5.

## 2. Fits

- **Splits:** step 1's six seed-0 splits, with step 1's fit carves and basis.
- **The screen's fits are this run's L-musique and L-hotpotqa fits** (`outputs/screen/fits/scr-qdepth` and
  `scr-qdepth-hp`, or `scr-relqd` and `scr-relqd-hp`). The run trains four fits, in the order L-2wiki, L-squad, J5,
  L-metaqa, through `outputs/mp_unified/qdepth.py train --split S --name S --arm A --out-root outputs/full_A/fits`,
  A being qdepth or relqd.
- **One fit per split:** variant p, seed 0, config 2e-3:1e-4:0.1:8:2, hidden 128, the SWA state p@swa. No select
  carves, no pick.
- **The prior:** U (1536 × 5), W (6 × 5) and b (5), zero at the start: 7,715 parameters beside the base model's.
- **relqd on L-metaqa is qdepth on step 1's blocks.** That split trains no KB, so rel's blocks are 0 on every training
  row and lean_gpu's dead-block rule drops them, as in rel's own L-metaqa fit. That fit then compares the prior with
  step 1's model on L-metaqa.
- **On the card:** qdepth takes step 1's caps, 0.26 (share 0.28) for J5, L-squad and L-2wiki and 0.18 (0.20) for
  L-metaqa; relqd takes rel's, 0.30 (0.32) and 0.22 (0.24). Reads 0.28 (0.30). The prior adds no columns.

## 3. Reads and comparison

- Each fit is read on the six s1eval carves.
- Each read is compared, question by question, with the base's fit of the same split, p@swa, on the same carves.
  That comparison decides.
  - qdepth: step 1's fit, `outputs/step1/fits/S`.
  - relqd: rel's fit, `outputs/full_rel/fits/S` (rel's screen fits for L-musique and L-hotpotqa), with step 1's
    reported beside it.
  - The comparison is lean_screen's: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS
    or WITHIN with the floor 0.0075.
  - Twenty-four reads come from these fits and twelve from the screen: 36 in all.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot), and each leave-out
  fit on its held-out dataset.

## 4. Verdict

- **qdepth:** `lean_screen2.py grade --grade-arm qdepth --full-root outputs/full_qdepth --reuse
  L-musique=outputs/screen/scr-qdepth.json,L-hotpotqa=outputs/screen/scr-qdepth-hp.json` files the grade.
- **relqd:** `qdepth.py grade-rel --full-root outputs/full_relqd --reuse
  L-musique=outputs/screen/scr-relqd.json,L-hotpotqa=outputs/screen/scr-relqd-hp.json` files it: relz.py's grade,
  decided against rel's fits, under relqd's name.
- **The re-grade decides.** `nullx.py regrade` re-grades all 36 reads under the seed null over every split into
  `grade-nullx.{md,json}` beside the grade. For relqd, each read's base R@5 is rel's (`--base-compares`, rel's six
  comparisons with step 1's fits).
- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

**After ADOPT,** the arm becomes the base of every later screen and run.

**Every fit runs to its end.** A LOSS filed at the floor 0.0075 may turn WITHIN under the null, so no fit is stopped
early.

## 5. Gates and order of work (8 October)

- **`scr11-base-step1` and `scr11-base-rel`** (`qdepth.py base`) run when rel's re-grade lands, about 06:15 to 06:30.
  The branch whose gate exits 1 is dropped by the feeder, with every item behind it.
- **`fqd-gate` and `frq-gate`** (`screen_gate.py pass` on the base's re-call) exit 0 only when the re-call is
  PROMISING. The four fits wait on it. On any other result the feeder drops them with their reads, comparisons and
  grades.
- **The items** go at the end of the feeder's list, after relz's full run.
- **ETAs if it passes:** the re-call lands about 07:15 to 07:45. The fits take about 45 to 60 minutes each beside the
  others. The grade comes about 09:30 to 10:30.

## 6. What this does not do

- No new carves, seeds, variants, columns or hyperparameters. The prior is one learned term over WALK's columns and the
  question's embedding, which every fit already reads.
- No new graph, text, encoder or model. The encoder and the substrate embeddings stay frozen.
- webqsp never trains. Test splits are never read.

## Results
