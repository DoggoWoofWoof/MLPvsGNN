# Full runs: rank inputs (prank) and gradient surgery (gsurg) on all six splits, each only if its screen is PROMISING

Declared 7 October 2026 at about 21:15, before any number of round five's screens exists (docs/SCREENS.md, fifth
round). Each arm's run starts only if its screen's verdict over both fits (`scr-prank-pair`, `scr-gsurg-pair`) is
PROMISING by section 2's rule. Otherwise none of that arm's run happens, and this file records why. Numbers here are
development numbers; the paper's numbers come from one declared confirmation run.

## 1. Question

- **prank** replaces each fixed block's within-pool z-score with the column's rank in the pool from the top and from
  the bottom, each as 60 / (60 + min(rank, 50)). The raw values, the presence flags, SEMB's z-score and rrf's base
  z-score stay as in step 1.
- **gsurg** trains step 1's model with lean_gpu's loop. The only change: before each step, the step's gradient loses
  its component along any other training dataset's gradient in the same chunk that it conflicts with (PCGrad).

Both are label-free and add no hyperparameter. prank changes every read alike; gsurg changes no read.

For an arm whose screen is PROMISING: does the gain hold on every split, and does nothing else lose? The splits are
J5 (in-domain on all five training datasets) and each leave-one-out fit on its held-out dataset. The screens read
musique, hotpotqa and webqsp zero-shot. Only these runs read metaqa, squad and 2wiki zero-shot, and every dataset
in-domain under J5.

## 2. Fits

- **Splits:** the six seed-0 splits of step 1: J5 and the leave-one-out fits L-metaqa, L-squad, L-musique,
  L-hotpotqa and L-2wiki. Each split's fit carves and basis are step 1's; the second-round smoke (`scr2-smoke`, passed)
  checked them against step 1's train.json.
- **The screens' fits are this run's L-musique and L-hotpotqa fits** (`outputs/screen/fits/scr-<arm>` and
  `scr-<arm>-hp`): the same arm, carves, config and seed. The null check showed the harness reproduces step 1 bit for
  bit, so a second fit would equal them. Each run trains four fits, in the order L-2wiki, L-squad, J5, L-metaqa.
  L-2wiki goes first because 2wiki read zero-shot is the read no screen makes.
- **One fit per split:** variant p, seed 0, config 2e-3:1e-4:0.1:8:2, hidden 128, the SWA state p@swa. No select
  carves, no pick: the screen's rule (docs/SCREENS.md).
- **Arms:** lean_screen5.py's prank and gsurg, trained through `outputs/mp_unified/lean_screen5.py train --split`, into
  `outputs/full_prank/fits` and `outputs/full_gsurg/fits`.
- **On the card:** zret's and ztop50's caps. J5, L-squad and L-2wiki cap 0.26 of the card (share 0.28); L-metaqa caps
  0.18 (share 0.20).
  - prank adds sorts over 16 columns at a time (tens of MB against about 5 GB).
  - gsurg keeps a chunk's batches together: the same 32 questions, one extra forward and backward at a time.

## 3. Reads and comparison

- Each fit is read on the six s1eval carves.
- Each read is compared, question by question, with step 1's fit of the same split, p@swa, on the same carves.
  - That is lean_screen's compare: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS or
    WITHIN with the floor 0.0075.
  - Twenty-four reads come from these fits and twelve from the screens: 36 in all.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot), and each
  leave-out fit on its held-out dataset.

## 4. Verdict (`lean_screen2.py grade --grade-arm prank` or `--grade-arm gsurg`)

- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

**After ADOPT,** the arm becomes the base of every later screen and run. A later screen's arm is added on top of it,
and its baseline becomes the adopted L-musique and L-hotpotqa fits. Step 1's fits stay the reference for everything
already read.

**If both arms are ADOPTED,** one changes the inputs and the other the objective, so they combine. A screen of the two
together decides the next base. It is declared in docs/SCREENS.md before its numbers.

**The first LOSS settles NOT_ADOPTED.** That arm's fits still running are then stopped, and the ones not yet started
are taken out of the queue, since no read they could give changes the verdict.

## 5. Gates and order of work (7 October)

- **`fpr-gate` and `fgs-gate`** (`outputs/mp_unified/screen_gate.py pass`, unchanged) run when the screen's pair
  verdict is filed (`outputs/screen/scr-prank-pair.json`, `scr-gsurg-pair.json`, through screen_pair.py).
  - Each exits 0 only when its pair is PROMISING, and its arm's four fits wait on it.
  - On any other verdict the feeder drops them with their reads, comparisons and grade.
- **No hold on the seed-0 GNN chain.** Its last items end about 22:30, before either gate can run.
- **ETAs if PROMISING:**
  - prank: the pair lands about 23:15 to 23:30. Its fits take about 25 to 35 minutes each, three at a time on the
    card, so the grade comes about 00:30 to 01:00.
  - gsurg: the pair lands about 23:40 to 00:15. Its fits take about 50 to 80 minutes each. J5 trains five datasets,
    so each of its steps adds four passes, and it takes longest. The grade comes about 01:30 to 02:30.
  - Both grades come sooner at a LOSS.

- **Amended 8 October, about 00:10, before any of the seed null's numbers** (docs/SCREENS.md, section 2): the
  screen's pair is re-called under the null's floors, and the re-call decides.
  - If `scr-gsurg-pair-recall` (or `scr-prank-pair-recall`) is PROMISING, that arm's run is re-queued as declared
    above, under new item names: `fgs-*-rc` behind `fgs-gate-rc` (or `fpr-*-rc` behind `fpr-gate-rc`), which runs
    `screen_gate.py pass` on the re-call.
  - The grade's twelve reused screen reads take the null's floors: `fgs-grade-rc-r` (`fpr-grade-rc-r`) re-calls
    the grade into `outputs/full_gsurg/grade-recall.{md,json}` (`outputs/full_prank/...`), and the re-grade
    decides.
  - On any other re-call verdict nothing here runs. The re-calls land about 01:15 to 01:30.

## 6. What this does not do

- No new carves, seeds or variants.
- prank's constants (60, from rank fusion; 50, ztop50's reference size) are fixed here, before any number, and are not
  tuned. gsurg has no constant.
- No new graph, text, encoder or model: both arms work on the graphs and features step 1 trains on. The encoder and the
  substrate embeddings stay frozen.
- webqsp never trains. Test splits are never read.

## Results

Development numbers; the paper's numbers come from one declared confirmation run.

### gsurg: not run (23:08)

Its screen pair is MIXED (`outputs/screen/scr-gsurg-pair.md`; docs/SCREENS.md, Results). The L-musique fit is
PROMISING (musique read zero-shot +0.0842, metaqa in-domain +0.0012), but in the L-hotpotqa fit hotpotqa, read
zero-shot, LOSEs (−0.0147). `fgs-gate` fails, and the feeder drops the four fits, their reads and comparisons, and
the grade.

### prank: not run (00:08)

Its screen pair is MIXED (`outputs/screen/scr-prank-pair.md`; docs/SCREENS.md, Results). musique read zero-shot
gains 0.2016 and webqsp read zero-shot 0.0269 in the L-musique fit. metaqa in-domain LOSEs in both fits (−0.0310,
−0.0320), as do hotpotqa in-domain in the L-musique fit (−0.0080) and hotpotqa read zero-shot in the L-hotpotqa fit
(−0.0255). `fpr-gate` fails and the feeder drops the four fits, their reads and comparisons, and the grade. The
re-queued run (`fpr-*-rc`, section 5) starts only if the seed null's re-call turns the pair PROMISING.
