# Full run: hub-discounted walks (hubwalk) on all six splits, only if its screen is PROMISING

Declared 7 October 2026 at about 21:25, before any number of the hubwalk screen exists (docs/SCREENS.md, sixth round).
It runs only if that screen's verdict over both its fits (`scr-hubwalk-pair`) is PROMISING by section 2's rule.
Otherwise nothing here runs, and this file records why. Numbers here are development numbers; the paper's numbers
come from one declared confirmation run.

## 1. Question

hubwalk replaces every walk column the model reads with the walk's probability mass:
- A walk starts uniform on its seeds.
- At each hop, each node's mass splits evenly over its out-edges in that walk's edge set.
- A column is log1p(n × mass), the lift over the pool's uniform distribution.

The degree column, the first-hop flags, the seed flag and every other block stay step 1's bit for bit. It is
label-free and changes every training dataset and every read alike.

If its screen is PROMISING, does the gain hold on every split, and does nothing else lose? The splits are J5
(in-domain on all five training datasets) and each leave-one-out fit on its held-out dataset. The screen reads
musique, hotpotqa and webqsp zero-shot. Only this run reads metaqa, squad and 2wiki zero-shot, and every dataset
in-domain under J5.

## 2. Fits

- **Splits:** the six seed-0 splits of step 1: J5 and the leave-one-out fits L-metaqa, L-squad, L-musique,
  L-hotpotqa and L-2wiki. Each split's fit carves and basis are step 1's.
- **The screen's two fits are this run's L-musique and L-hotpotqa fits** (`outputs/screen/fits/scr-hubwalk` and
  `scr-hubwalk-hp`): the same arm, carves, config and seed. The run trains four fits, in the order L-2wiki, L-squad,
  J5, L-metaqa. L-2wiki goes first because 2wiki read zero-shot is the read no screen makes.
- **The arrays:** the screen's build covers every carve these fits train on (metaqa, squad, hotpotqa and 2wiki fit;
  musique s1fit) and read (the six s1eval carves). No new build.
- **One fit per split:** variant p, seed 0, config 2e-3:1e-4:0.1:8:2, hidden 128, the SWA state p@swa. No select
  carves, no pick: the screen's rule (docs/SCREENS.md).
- **Arm:** hubwalk.py's hubwalk, trained through `outputs/mp_unified/hubwalk.py train --split`, into
  `outputs/full_hubwalk/fits`.
- **On the card:** the earlier runs' caps. The arm changes only what the carve holds, not its size. J5, L-squad and
  L-2wiki cap 0.26 of the card (share 0.28); L-metaqa caps 0.18 (share 0.20).

## 3. Reads and comparison

- Each fit is read on the six s1eval carves.
- Each read is compared, question by question, with step 1's fit of the same split, p@swa, on the same carves.
  - That is lean_screen's compare: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS or
    WITHIN with the floor 0.0075.
  - Twenty-four reads come from these fits and twelve from the screen: 36 in all.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot), and each
  leave-out fit on its held-out dataset.

## 4. Verdict (`lean_screen2.py grade --grade-arm hubwalk`)

- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

**After ADOPT,** hubwalk becomes the base of every later screen and run. Its arrays become the walk features every
later fit reads, and a later screen's baseline becomes the adopted L-musique and L-hotpotqa fits. If round five's prank
or gsurg is also ADOPTED, a screen of the adopted arms together decides the next base. It is declared in
docs/SCREENS.md before its numbers.

**The first LOSS settles NOT_ADOPTED.** The fits still running are then stopped, and the ones not yet started are taken
out of the queue.

## 5. Gates and order of work (7 October)

- **`fhw-gate`** (`outputs/mp_unified/screen_gate.py pass`, unchanged) runs when the pair verdict is filed
  (`outputs/screen/scr-hubwalk-pair.json`, through screen_pair.py).
  - It exits 0 only when that verdict is PROMISING, and the four fits wait on it.
  - On any other verdict the feeder drops them with their reads, comparisons and grade.
- **ETA if PROMISING:** the pair lands about 23:45 to 00:30. The fits take about 25 to 35 minutes each, three at a time
  on the card, so the grade comes about 01:15 to 02:00, sooner at a LOSS.

## 6. What this does not do

- No new carves, seeds or variants. The walk's form (uniform start, even split, lift over uniform) is fixed here,
  before any number, and is not tuned.
- No new graph, text, encoder or model: the arm reads the same query graphs, edge families and seeds as step 1. The
  encoder and the substrate embeddings stay frozen.
- webqsp never trains. Test splits are never read.

## Results

Development numbers; the paper's numbers come from one declared confirmation run.
