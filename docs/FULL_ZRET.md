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

**16:50: the screen is PROMISING, and the run started by itself at 16:47** (J5, L-metaqa and L-squad first). zret's
epochs take about twice the base's (the screen's fit: 43 minutes). The fits take about 50 to 70 minutes each, three
at a time on the card. J5's and L-metaqa's comparisons come about 18:00; the verdict about 18:45 to 19:15, sooner if
a comparison shows a LOSS.

## 6. What this does not do

- No new carves, seeds or variants.
- The encoder and the substrate embeddings stay frozen.
- webqsp never trains. Test splits are never read.

## Results

Development numbers; the paper's numbers come from one declared confirmation run.

### NOT_ADOPTED (grade filed 17:57)

`outputs/full_zret/grade.md`. R@5 of zret's p@swa minus step 1's p@swa of the same split, on the six s1eval
carves, with the call by docs/SCREENS.md's rule. Bold: step 1's eleven primary reads. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **−0.0011 WITHIN** | **+0.0013 WITHIN** | **+0.0031 WITHIN** | **−0.0005 WITHIN** | **−0.0035 WITHIN** | **+0.0250 GAIN zs** |
| L-metaqa | **+0.0644 GAIN zs** | +0.0029 WITHIN | +0.0087 GAIN | −0.0032 WITHIN | +0.0066 WITHIN | +0.0071 WITHIN zs |
| L-squad | −0.0063 WITHIN | **+0.0056 WITHIN zs** | −0.0093 LOSS | −0.0017 WITHIN | −0.0048 WITHIN | +0.0166 GAIN zs |
| L-musique | −0.0041 WITHIN | +0.0007 WITHIN | **+0.1202 GAIN zs** | +0.0046 WITHIN | −0.0018 WITHIN | −0.0012 WITHIN zs |
| L-hotpotqa | −0.0121 LOSS | +0.0011 WITHIN | +0.0073 WITHIN | **−0.0140 LOSS zs** | +0.0056 WITHIN | −0.0005 WITHIN zs |
| L-2wiki | −0.0004 WITHIN | −0.0003 WITHIN | +0.0055 WITHIN | +0.0058 WITHIN | **−0.0081 LOSS zs** | +0.0262 GAIN zs |

Three of the eleven primary reads GAIN, and four of the thirty-six reads LOSE. The L-musique row is the screen's fit
(`--reuse`).

- **It gains where the pools are big and thinly retrieved.** All of these are zero-shot reads. metaqa from L-metaqa
  rises from 0.077 to 0.142 (+0.064 [+0.059, +0.070]). musique from L-musique rises from 0.270 to 0.390. webqsp from
  J5 rises from 0.117 to 0.142 (+0.025), and it gains in L-squad's and L-2wiki's fits too (+0.017, +0.026).
- **It loses where the pools are small.** hotpotqa read zero-shot falls from 0.859 to 0.845 (−0.014 [−0.018, −0.010];
  FC@5 −0.026). 2wiki read zero-shot falls from 0.807 to 0.799 (−0.008 [−0.011, −0.005]; FC@5 −0.018). Two in-domain
  reads also lose: metaqa in L-hotpotqa's fit (−0.012) and musique in L-squad's fit (−0.009).
- **J5 is within the floor on all five in-domain datasets** (−0.0035 to +0.0031).
- **The screen could not see the losses.** In L-musique's fit, hotpotqa and 2wiki are training datasets; only L-hotpotqa
  and L-2wiki read them zero-shot. Block dropout's full run failed the same way, on reads its screen did not make
  (docs/FULL_BDROP20.md).
- **What zret left in.** The retrieved shares are 0.10 to 0.18 in the big pools (metaqa, webqsp, musique) and 0.56 to
  0.69 in hotpotqa's and 2wiki's (docs/SCREENS.md, pool composition). zret took the share out of the z-scores but kept
  the number of retrieved rows they are taken over: 205 to 379 in the big pools, 50 to 63 in the small ones. Over
  more rows, a pool's best rows sit further out in the tail. The fourth round (ztop50, docs/SCREENS.md) tests whether
  that number is the channel left.
- **Nothing was stopped early.** The first LOSS (L-squad's comparison, 17:47) came while the laptop could not reach
  the host (a laptop-side Tailscale fault, 17:06 to 19:24). L-hotpotqa and L-2wiki were not stopped; their
  comparisons landed at 17:54 and 17:56. The grade released the seed-0 GNN chain at 17:57, as section 5 declared.
- zret is not the base of later screens. Step 1's fits stay the base.

### Re-graded under the seed null over every split (declared 8 October about 02:45, before its numbers)

docs/SCREENS.md, section 2. zret's grade is complete, so `nullx.py regrade` re-grades its 36 reads with the floors of
the null over all six splits, into `outputs/full_zret/grade-nullx.{md,json}`, when the null lands (about 04:30 to
05:30). Its filed LOSSes (L-squad musique −0.0093, L-hotpotqa metaqa −0.0121 and hotpotqa read zero-shot −0.0140,
L-2wiki 2wiki read zero-shot −0.0081) and its GAINs (L-metaqa metaqa read zero-shot +0.0644 among them) each meet
their read's floor. The re-grade is filed; what an ADOPT would change is declared in a later round.
