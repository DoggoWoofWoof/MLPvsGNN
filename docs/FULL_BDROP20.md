# Full run: block dropout in training (bdrop20), on all six splits

Declared 7 October 2026 at about 15:20, before any number of this run exists. Its screen (docs/SCREENS.md,
scr-bdrop20) was PROMISING. Numbers here are development numbers; the paper's numbers come from one declared
confirmation run.

## 1. Question

In training, each (question, feature block) pair is kept with probability 0.8. A dropped block's columns and keep
flag are 0, the model's input for a missing block. Every block is present at read time.

On the screen split (L-musique), this raised musique, read zero-shot, by 0.134 R@5 (0.270 to 0.404; rrf 0.473), and
no read lost.

Does it hold on every split: in-domain on all five training datasets (J5), and zero-shot on each held-out dataset?

## 2. Fits

- **Splits:** the six seed-0 splits of step 1: J5 and the leave-one-out fits L-metaqa, L-squad, L-musique,
  L-hotpotqa and L-2wiki. Each split's fit carves and basis are step 1's. The smoke (`scr2-smoke`) checks them
  against step 1's train.json.
- **One fit per split:** variant p, seed 0, config 2e-3:1e-4:0.1:8:2, hidden 128, the SWA state p@swa. There are no
  select carves and no pick; this is the screen's rule (docs/SCREENS.md).
- **Arm:** lean_screen.py's bdrop20 (P = 0.2, its own generator at seed + 101), trained through
  `outputs/mp_unified/lean_screen2.py train --split`.
- **L-musique's fit is the screen's** (`outputs/screen/fits/scr-bdrop20`): the same arm, carves, config and seed. The
  null check showed the harness reproduces step 1 bit for bit, so a second L-musique fit would equal it.
- **On the card:** caps from measured peaks. Variant p alone peaks at the fit's on-card data plus about 0.7 GB: J5
  about 5.4 GB, L-squad, L-hotpotqa and L-2wiki about 5.3 GB, L-metaqa about 3.2 GB. Caps are 0.26 of the card
  (share 0.28) for those four and 0.18 (share 0.20) for L-metaqa.

## 3. Reads and comparison

- Each fit is read on the six s1eval carves.
- Each read is compared, question by question, with step 1's fit of the same split, p@swa, on the same carves. This
  is lean_screen's compare: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS or WITHIN
  with the floor 0.0075. That makes 36 reads.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot), and each
  leave-out fit on its held-out dataset.

## 4. Verdict (`lean_screen2.py grade`)

- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

After ADOPT, block dropout 0.2 joins the training of every later screen and run. A later screen's arm is added on
top of it, and its baseline becomes the adopted L-musique fit (scr-bdrop20). Step 1's fits stay the reference for
everything already read.

## 5. Order of work and ETAs (7 October)

1. Commit this file and `lean_screen2.py` (its selftest passes). Push.
2. On the host, after the smoke (about 15:25 to 15:35): the five fits on the card beside the second-round screens,
   then each fit's read and comparison.
3. The grade, when the five comparisons exist.

Each fit takes about 25 to 40 minutes on its own, longer while it shares the card. About four fit at once, so the
fits finish in two waves. Expected verdict: about 17:00 to 17:30.

## 6. What this does not do

- No new carves, seeds or variants.
- The encoder and the substrate embeddings stay frozen.
- webqsp never trains. Test splits are never read.

## Results

None yet.
