# Full runs: question-relation inputs (rel) and retrieval propagation (gcs) on all six splits, each only if its screen's re-call is PROMISING

Declared 8 October 2026 at about 01:00, before any number of round nine's screens exists (docs/SCREENS.md, ninth
round). Each arm's run starts only if its screen pair's re-call under the seed null (`scr-rel-pair-recall`,
`scr-gcs-pair-recall`; docs/SCREENS.md, section 2) is PROMISING. Otherwise none of that arm's run happens, and this
file records why. Numbers here are development numbers; the paper's numbers come from one declared confirmation run.

## 1. Question

- **rel** adds three blocks of the look's compiled columns, 26 in all, after step 1's nine: typed_rel (10), typed_v2
  (6) and ordered (10). Each is a match between the question and the text of the relations around a node, on the
  paths from the seeds, or along the best typed walk. On a graph without typed relations every one is 0.
- **gcs** adds two columns: retrieval's scores spread two steps over the pool graph (over every edge family, and over
  the structural edges), GraphER's parameter-free propagation.

Both are label-free and the same in training and at read time. They add inputs only: the loop, batches, loss and
every other block are step 1's.

For an arm whose re-call is PROMISING: does the gain hold on every split, and does nothing else lose? The splits are J5
(in-domain on all five training datasets) and each leave-one-out fit on its held-out dataset. The screens read
musique, hotpotqa and webqsp zero-shot. Only these runs read metaqa, squad and 2wiki zero-shot, and every dataset
in-domain under J5.

## 2. Fits

- **Splits:** the six seed-0 splits of step 1: J5 and the leave-one-out fits L-metaqa, L-squad, L-musique,
  L-hotpotqa and L-2wiki. Each split's fit carves and basis are step 1's.
- **The screens' fits are this run's L-musique and L-hotpotqa fits** (`outputs/screen/fits/scr-<arm>` and
  `scr-<arm>-hp`): the same arm, carves, config and seed. Each run trains four fits, in the order L-2wiki, L-squad,
  J5, L-metaqa. L-2wiki goes first because 2wiki read zero-shot is the read no screen makes.
- **One fit per split:** variant p, seed 0, config 2e-3:1e-4:0.1:8:2, hidden 128, the SWA state p@swa. No select
  carves, no pick: the screen's rule (docs/SCREENS.md).
- **Arms:** relcols.py's rel and gcs, trained through `outputs/mp_unified/relcols.py train --split`, into
  `outputs/full_rel/fits` and `outputs/full_gcs/fits`.
- **rel on L-metaqa is step 1's model.** That split trains no KB, so the rel blocks are 0 on every training row, and
  lean_gpu's dead-block rule drops them. The fit still runs, and its six reads must come out with a difference of
  exactly 0. Any other result means the harness no longer reproduces step 1, and this file says so.
- **On the card,** sized by the measured peaks (step 1's L-musique fit 4.5 GB, L-metaqa 4.2 GB; L-hotpotqa's screen
  fits 5.2 GB):
  - **rel** loads its 26 columns as float16 beside step 1's: about 1.2 GB more on J5, L-squad and L-2wiki, and 0.6 GB
    on L-metaqa, where the blocks are loaded and then dropped. J5, L-squad and L-2wiki cap 0.30 of the card (share
    0.32), L-metaqa 0.22 (share 0.24).
  - **gcs** adds two columns, under 0.1 GB, and keeps the earlier runs' caps: J5, L-squad and L-2wiki 0.26 (share
    0.28), L-metaqa 0.18 (share 0.20).
  - Reads cap 0.28 (share 0.30). rel's largest read carve, metaqa s1eval, adds about 1.0 GB to the measured 4.9 GB.

## 3. Reads and comparison

- Each fit is read on the six s1eval carves.
- Each read is compared, question by question, with step 1's fit of the same split, p@swa, on the same carves.
  - That is lean_screen's compare: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS or
    WITHIN with the floor 0.0075.
  - Twenty-four reads come from these fits and twelve from the screens: 36 in all.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot), and each
  leave-out fit on its held-out dataset.

## 4. Verdict

- `lean_screen2.py grade --grade-arm rel` (or `gcs`) files the grade, and `screen_recall.py grade` re-calls its twelve
  reused screen reads under the null's floors, into `outputs/full_rel/grade-recall.{md,json}` (or
  `outputs/full_gcs/...`). Its other 24 reads keep 0.0075, since no null covers their splits. **The re-grade decides.**
- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

**After ADOPT,** the arm becomes the base of every later screen and run. A later screen's arm is added on top of it,
and its baseline becomes the adopted L-musique and L-hotpotqa fits. Step 1's fits stay the reference for everything
already read.

**If both arms are ADOPTED,** they add different blocks and combine. A screen of the two together decides the next
base. It is declared in docs/SCREENS.md before its numbers.

**The first LOSS settles NOT_ADOPTED.** That arm's fits still running are then stopped, and the ones not yet started
are taken out of the queue, since no read they could give changes the verdict.

## 5. Gates and order of work (8 October)

- **`frl-gate-rc` and `fgc-gate-rc`** (`outputs/mp_unified/screen_gate.py pass`, unchanged) run when the pair's
  re-call is filed (`outputs/screen/scr-rel-pair-recall.json`, `scr-gcs-pair-recall.json`, through
  screen_recall.py).
  - Each exits 0 only when its re-call is PROMISING, and its arm's four fits wait on it.
  - On any other verdict the feeder drops them with their reads, comparisons and grades.
- **The pair verdicts** (`scr-rel-pair`, `scr-gcs-pair`) are filed and reported; nothing waits on them.
- **ETAs if PROMISING:** the re-calls land about 02:30 to 03:15. The fits take about 25 to 35 minutes each alone,
  about 50 with three on the card, and they queue behind every earlier full run that its gate lets through. The grades
  come about 04:30 to 06:00, sooner at a LOSS.

## 6. What this does not do

- No new carves, seeds or variants. The arms' columns are the look's compiled columns as the twin and the GNN read
  them; nothing in them is chosen or tuned here.
- No new graph, text, encoder or model. The relation texts are the KBs' own relation names, embedded by the frozen
  encoder when the looks were compiled. The encoder and the substrate embeddings stay frozen.
- webqsp never trains. Test splits are never read.

## Results

Development numbers; the paper's numbers come from one declared confirmation run.
