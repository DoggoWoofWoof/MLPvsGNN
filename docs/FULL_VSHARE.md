# Full run: share augmentation of the pool statistics (vshare) on all six splits, only if its screen is PROMISING

Declared 7 October 2026 at about 22:35, before any number of the vshare screen exists (docs/SCREENS.md, seventh
round). It runs only if that screen's verdict over both its fits (`scr-vshare-pair`) is PROMISING by section 2's rule.
Otherwise nothing here runs, and this file records why. Numbers here are development numbers; the paper's numbers
come from one declared confirmation run.

## 1. Question

vshare reweights a training pool's statistics, never its rows. With probability 0.5, a training question's pool
z-scores (every fixed block's, SEMB's and rrf's base z-score) are taken with row weights:
- The weights make retrieval's ranked rows a share t of the pool's weight.
- t is uniform in [0.08, 0.20] when the pool's own share is at least 0.375, and in [0.55, 0.95] when it is below.
- Ranked rows weigh 1; unranked rows weigh r(1 − t) / (t u), for r ranked and u unranked rows.

The listwise softmax, the raw values and the golds stay the pool's own, and reads are never reweighted. It is
label-free and changes every training dataset alike.

If its screen is PROMISING, does the gain hold on every split, and does nothing else lose? The splits are J5
(in-domain on all five training datasets) and each leave-one-out fit on its held-out dataset. The screen reads
musique, hotpotqa and webqsp zero-shot. Only this run reads metaqa, squad and 2wiki zero-shot, and every dataset
in-domain under J5.

## 2. Fits

- **Splits:** the six seed-0 splits of step 1: J5 and the leave-one-out fits L-metaqa, L-squad, L-musique,
  L-hotpotqa and L-2wiki. Each split's fit carves and basis are step 1's.
- **The screen's two fits are this run's L-musique and L-hotpotqa fits** (`outputs/screen/fits/scr-vshare` and
  `scr-vshare-hp`): the same arm, carves, config and seed. The run trains four fits, in the order L-2wiki, L-squad,
  J5, L-metaqa. L-2wiki goes first because 2wiki read zero-shot is the read no screen makes.
- **One fit per split:** variant p, seed 0, config 2e-3:1e-4:0.1:8:2, hidden 128, the SWA state p@swa. No select
  carves, no pick: the screen's rule (docs/SCREENS.md).
- **Arm:** lean_screen7.py's vshare, trained through `outputs/mp_unified/lean_screen7.py train --split`, into
  `outputs/full_vshare/fits`.
- **On the card:** the earlier runs' caps. The arm adds no memory beyond one weight per batch row. J5, L-squad and
  L-2wiki cap 0.26 of the card (share 0.28); L-metaqa caps 0.18 (share 0.20).

## 3. Reads and comparison

- Each fit is read on the six s1eval carves.
- Each read is compared, question by question, with step 1's fit of the same split, p@swa, on the same carves.
  - That is lean_screen's compare: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS or
    WITHIN with the floor 0.0075.
  - Twenty-four reads come from these fits and twelve from the screen: 36 in all.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot), and each
  leave-out fit on its held-out dataset.

## 4. Verdict (`lean_screen2.py grade --grade-arm vshare`)

- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

**After ADOPT,** vshare becomes the base of every later screen and run, and a later screen's baseline becomes the
adopted L-musique and L-hotpotqa fits. If another round's arm is also ADOPTED, a screen of the adopted arms together
decides the next base. It is declared in docs/SCREENS.md before its numbers.

**The first LOSS settles NOT_ADOPTED.** The fits still running are then stopped, and the ones not yet started are taken
out of the queue.

## 5. Gates and order of work (7 October)

- **`fvs-gate`** (`outputs/mp_unified/screen_gate.py pass`, unchanged) runs when the pair verdict is filed
  (`outputs/screen/scr-vshare-pair.json`, through screen_pair.py).
  - It exits 0 only when that verdict is PROMISING, and the four fits wait on it.
  - On any other verdict the feeder drops them with their reads, comparisons and grade.
- **ETA if PROMISING:** the pair lands about 00:30 to 01:15. The fits take about 25 to 35 minutes each alone, about
  50 with three on the card, so the grade comes about 02:00 to 03:00, sooner at a LOSS.

- **Amended 7 October, about 23:45, before the L-hotpotqa fit of the screen's numbers (the L-musique fit was PROMISING): the seed null** (docs/SCREENS.md, section 2).
  - The four fits now wait on `fvs-gate-r`, which runs `screen_gate.py pass` on the screen's re-call
    (`outputs/screen/scr-vshare-pair-recall.json`), in place of `fvs-gate`. The re-call is the pair verdict
    with each read's floor raised to twice the seed-only spread where that spread is larger than 0.0075.
  - `fvs-gate` still runs on the pair verdict and is reported; nothing waits on it.
  - The fits start only when the re-call is PROMISING, once the null lands: about 01:00 to 01:30 at the earliest.
  - The grade's twelve reused screen reads (L-musique, L-hotpotqa) are called under the null's floors. Its other 24
    reads keep 0.0075, since no null covers their splits.
  - That re-grade is `screen_recall.py grade` on the filed grade (item `fvs-grade-r`, after the grade), into
    `outputs/full_vshare/grade-recall.{md,json}`. **The re-grade decides** ADOPT or NOT_ADOPTED.

## 6. What this does not do

- No new carves, seeds or variants. The arm's form (probability 0.5, the two share ranges, the 0.375 split, the
  weights) is fixed here, before any number, and is not tuned.
- No new graph, text, encoder or model: the arm reads the same pools, features and seeds as step 1. The encoder and the
  substrate embeddings stay frozen.
- webqsp never trains. Test splits are never read.

## Results

Development numbers; the paper's numbers come from one declared confirmation run.
