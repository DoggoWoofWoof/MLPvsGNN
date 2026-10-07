# Full run: pool z-scores over the top 50 retrieved rows (ztop50), on all six splits, if its screen is PROMISING

Declared 7 October 2026 at about 19:55, before any number of the ztop50 screen exists. It runs only if that screen
(docs/SCREENS.md, scr-ztop50) is PROMISING by section 2's rule. Otherwise nothing here runs, and this file records
why. Numbers here are development numbers; the paper's numbers come from one declared confirmation run.

## 1. Question

ztop50 takes every pool z-score in the model against the pool's top 50 retrieved rows by rrf (rrf > 0; ties by row
order) instead of all of its rows. That covers each block's z-scores and rrf's base z-score. A pool with fewer than
two reference rows, or a column constant over them, keeps its whole-pool z-score there. The change is label-free,
the same in training and at read time, and adds no parameters. Where no pool has more than 50 retrieved rows it is
zret bit for bit; on squad's pools (50 rows, all retrieved) it is the base model bit for bit.

zret's full run (docs/FULL_ZRET.md) gained on the big pools read zero-shot and lost on the small ones. ztop50 keeps
zret's change and also takes out the number of retrieved rows a z-score is measured against (docs/SCREENS.md, fourth
round).

If its screen on L-musique is PROMISING, does the gain hold on every split, and does nothing else lose? The splits are
J5 (in-domain on all five training datasets) and each leave-one-out fit on its held-out dataset. The screen cannot
read hotpotqa or 2wiki zero-shot, where zret lost. Only this run reads them.

## 2. Fits

- **Splits:** the six seed-0 splits of step 1: J5 and the leave-one-out fits L-metaqa, L-squad, L-musique,
  L-hotpotqa and L-2wiki. Each split's fit carves and basis are step 1's; the second-round smoke (`scr2-smoke`, passed)
  checked them against step 1's train.json.
- **One fit per split:** variant p, seed 0, config 2e-3:1e-4:0.1:8:2, hidden 128, the SWA state p@swa. No select
  carves, no pick: the screen's rule (docs/SCREENS.md).
- **Arm:** lean_screen4.py's ztop50, trained through `outputs/mp_unified/lean_screen4.py train --split`.
- **L-musique's fit is the screen's** (`outputs/screen/fits/scr-ztop50`): the same arm, carves, config and seed. The
  null check showed the harness reproduces step 1 bit for bit, so a second L-musique fit would equal it.
- **On the card:** zret's caps. ztop50 adds two sorts per batch (tens of MB against about 5 GB). J5, L-squad,
  L-hotpotqa and L-2wiki cap 0.26 of the card (share 0.28); L-metaqa caps 0.18 (share 0.20).
- **Order:** L-hotpotqa and L-2wiki first, where zret lost read zero-shot; then L-squad, J5 and L-metaqa.
- **Amended 7 October, about 20:35, before any ztop50 number (docs/SCREENS.md section 2: two fits per screen):**
  L-hotpotqa's fit is now the screen's second fit (`outputs/screen/fits/scr-ztop50-hp`): the same arm, carves, config,
  seed and caps as declared here. The full run trains four fits, in the order L-2wiki, L-squad, J5, L-metaqa. The grade
  reuses both screen comparisons: L-musique from `outputs/screen/scr-ztop50.json` and L-hotpotqa from
  `outputs/screen/scr-ztop50-hp.json`.

## 3. Reads and comparison

- Each fit is read on the six s1eval carves.
- Each read is compared, question by question, with step 1's fit of the same split, p@swa, on the same carves. That
  is lean_screen's compare: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS or
  WITHIN with the floor 0.0075. That makes 36 reads.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot), and each
  leave-out fit on its held-out dataset.
- zret's full-run fits are not a base here. Its numbers are set beside this run's in the results, for the reader only.

## 4. Verdict (`lean_screen2.py grade --grade-arm ztop50`)

- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

After ADOPT, ztop50 becomes the base model of every later screen and run. A later screen's arm is added on top of it,
and its baseline becomes the adopted L-musique fit (scr-ztop50). Step 1's fits stay the reference for everything
already read.

The first LOSS settles NOT_ADOPTED. The fits still running then are stopped and the ones not yet started are taken
out of the queue, since no read they could give changes the verdict.

## 5. Gates and order of work (7 October)

- **`fzt-gate`** (`outputs/mp_unified/screen_gate.py pass`, unchanged) runs when the screen's comparison
  (`scr-compare-ztop50`) is filed. It exits 0 only when scr-ztop50 is PROMISING. The five fits wait on it. On any
  other verdict the feeder drops them with their reads, comparisons and grade.
- **No hold on the seed-0 GNN chain this time.** It runs one item at a time at half the card, and its last twelve
  items end about 22:30. Beside it, two of these fits fit on the card (0.28 and 0.20); after it, three. Holding it, as
  zret's run did, would bring this verdict no sooner than about half an hour, and would leave the chain for after
  midnight.

- **Amended about 20:35:** `fzt-gate` now runs on the screen's verdict over both fits
  (`outputs/screen/scr-ztop50-pair.json`, filed by `scr-pair-ztop50` through screen_pair.py) and exits 0 only when that
  verdict is PROMISING. The pair verdict lands about 22:00 to 22:15. The four fits then take about 50 to 70 minutes
  each, two or three at a time on the card, so the grade comes about 00:00 to 01:00, sooner at a LOSS.

ETA if PROMISING (as declared at 19:55, before the amendment): the screen's comparison lands about 21:00 to 21:15. The fits take about 50 to 70 minutes each.
L-hotpotqa's and L-2wiki's comparisons come about 22:15 to 22:45; the verdict about 23:30 to 00:30, sooner at a
LOSS.

## 6. What this does not do

- No new carves, seeds or variants. The reference size (50) is fixed here, before any number, and is not tuned.
- The encoder and the substrate embeddings stay frozen.
- webqsp never trains. Test splits are never read.

## Results

Development numbers; the paper's numbers come from one declared confirmation run.

**The screen's L-musique fit is MIXED (20:39; docs/SCREENS.md, Results).** metaqa in-domain LOSEs (−0.0176),
while musique and webqsp read zero-shot GAIN (+0.139, +0.0175). One LOSS among the pair's twelve reads rules out
PROMISING, so `fzt-gate` fails when the pair verdict is filed (about 22:00) and the four fits do not run. ztop50 is
not adopted, and no later screen builds on it.

**Not run (21:25).** The screen's pair verdict is MIXED (`outputs/screen/scr-ztop50-pair.md`): GAINs on 3 of the
12 reads, LOSSes on 4. The L-hotpotqa fit lost metaqa in-domain (−0.025), hotpotqa read zero-shot (−0.008) and
webqsp read zero-shot (−0.017), and gained musique in-domain (+0.012). `fzt-gate` exited 1 and the feeder dropped
the four fits, their reads, comparisons and the grade. ztop50 is NOT_ADOPTED without a full run.
