# Full run: pool z-scores over retrieved rows (zret), on all six splits, if its screen is PROMISING

Declared 7 October 2026 at about 16:15, before any number of the zret screen exists. It runs only if that screen
(docs/SCREENS.md, scr-zret) is PROMISING by section 2's rule. Otherwise nothing here runs, and this file records why.
Numbers here are development numbers; the paper's numbers come from one declared confirmation run.

## 1. Question

zret takes every pool z-score in the model against the pool's retrieved rows (rrf > 0) instead of all of its rows.
That covers each block's z-scores and rrf's base z-score. A pool with fewer than two retrieved rows, or a column
constant over them, keeps its whole-pool z-score there. The change is label-free, the same in training and at read
time, and adds no parameters. Where every row is retrieved (squad's pools), it is the base model bit for bit.

If its screen on L-musique is PROMISING, does the gain hold on every split, and does nothing else lose? The splits are
J5 (in-domain on all five training datasets) and each leave-one-out fit on its held-out dataset.

Block dropout's screen passed, and its full run failed (docs/FULL_BDROP20.md). An L-musique gain can be specific to
musique; this run is what catches that.

## 2. Fits

- **Splits:** the six seed-0 splits of step 1: J5 and the leave-one-out fits L-metaqa, L-squad, L-musique,
  L-hotpotqa and L-2wiki. Each split's fit carves and basis are step 1's; the second-round smoke (`scr2-smoke`, passed)
  checked them against step 1's train.json.
- **One fit per split:** variant p, seed 0, config 2e-3:1e-4:0.1:8:2, hidden 128, the SWA state p@swa. No select
  carves, no pick: the screen's rule (docs/SCREENS.md).
- **Arm:** lean_screen3.py's zret, trained through `outputs/mp_unified/lean_screen3.py train --split`.
- **L-musique's fit is the screen's** (`outputs/screen/fits/scr-zret`): the same arm, carves, config and seed. The null
  check showed the harness reproduces step 1 bit for bit, so a second L-musique fit would equal it.
- **On the card:** bdrop20's measured caps. zret adds a few small per-batch tensors (tens of MB against about 5 GB).
  J5, L-squad, L-hotpotqa and L-2wiki cap 0.26 of the card (share 0.28); L-metaqa caps 0.18 (share 0.20). J5 and
  L-metaqa, where block dropout lost, are queued first.

## 3. Reads and comparison

- Each fit is read on the six s1eval carves.
- Each read is compared, question by question, with step 1's fit of the same split, p@swa, on the same carves. That
  is lean_screen's compare: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS or
  WITHIN with the floor 0.0075. That makes 36 reads.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot), and each
  leave-out fit on its held-out dataset.

## 4. Verdict (`lean_screen2.py grade --grade-arm zret`)

- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

After ADOPT, zret becomes the base model of every later screen and run. A later screen's arm is added on top of it,
and its baseline becomes the adopted L-musique fit (scr-zret). Step 1's fits stay the reference for everything
already read.

As with block dropout, the first LOSS settles NOT_ADOPTED. The fits still running then are stopped, since no read
they could give changes the verdict.

## 5. Gates and order of work (7 October)

Two CPU items on the host, both from `outputs/mp_unified/screen_gate.py` (its selftest passes), run when the screen's
comparison (`scr-compare-zret`) is filed:

- **`fz-gate`** exits 0 only when scr-zret is PROMISING. The five fits wait on it. On any other verdict the feeder
  drops them with their reads, comparisons and grade.
- **`chain-gate-zret`** holds the 30 seed-0 GNN items of parts 10 and 11 (one chain, about 0.55 of the card at a
  time) until this run's grade is filed. A 3-hour timeout or a release flag (`outputs/screen/chain-go.flag`, made by
  hand if the run is stopped early) also releases them. On a verdict other than PROMISING, it releases them at once.
  Each GNN run holds 0.55 of the card for an hour or more. Started beside this run, one would leave room for one fit
  at a time.

ETA if PROMISING: the screen's comparison lands about 16:30. The five fits take about 25 to 40 minutes each, two
waves on the card. The verdict comes about 17:40 to 18:00.

## 6. What this does not do

- No new carves, seeds or variants.
- The encoder and the substrate embeddings stay frozen.
- webqsp never trains. Test splits are never read.

## Results

Development numbers; the paper's numbers come from one declared confirmation run.
