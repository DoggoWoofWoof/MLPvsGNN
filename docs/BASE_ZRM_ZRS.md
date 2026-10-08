# The next base if zrm is ADOPTED too: zrm's fits against zrs's, read by read

Declared 8 October 2026 at about 09:50, after zrs's re-grade (ADOPT, 09:38) and before zrm's re-grade exists. zrm's
last fit (J5) is being read; its grade lands in minutes. docs/FULL_ROUND14.md and docs/FULL_ROUND15.md, section 4:
"If zgs or zrm is ADOPTED too, each re-grade is filed and the next base is declared in a later round, before its
numbers." This is that round. Numbers here are development numbers; the paper's numbers come from one declared
confirmation run.

## 1. Question

zrs (docs/FULL_ROUND15.md) trains rmatch's match alone on zret's frozen fits. zrm (docs/FULL_ROUND14.md) trains the
match and zret's model together. If both are ADOPTED against zret's fits: which is the base of every later screen and
run?

## 2. The incumbent is zrs

- zrs was ADOPTED first.
- It changes less: zret's fits stay frozen, and the match is added on top. Every untyped read is zret's bit for bit.
- zrm moves the model's own weights. So it must beat zrs with nothing lost, read by read, to become the base.

## 3. What is compared

- Nothing trains or reads, and no fit folder is written.
- **zrm's six fits against zrs's,** split by split, p@swa on the six s1eval carves: 36 reads.
  - zrm's fits: `scr-zrm` and `scr-zrm-hp` (its screen fits) on L-musique and L-hotpotqa, and
    `outputs/full_zrm/fits/S` on the other four splits.
  - zrs's fits: `scr-zrs` and `scr-zrs-hp`, and `outputs/full_zrs/fits/S`.
  - zret's and step 1's fits of each split are reported beside these and decide nothing.
- The comparison is lean_screen's: the R@5 difference with a 2,000-resample question bootstrap, filed at the floor
  0.0075 (`outputs/zbase/compare-S`). The grade is relz.py's, decided against zrs's fit of each split
  (`outputs/zbase/grade`).
- **The re-grade decides.** `nullx.py regrade` calls every read under the seed null over every split
  (`outputs/zbase/grade-nullx`). Each read's base R@5 is zrs's: zrs's six fits compared with step 1's
  (`outputs/zbase/base-zrs-S`). Later rounds decided against zrs's fits reuse these.
- **Primary reads** are step 1's eleven: J5 on all six datasets, and each leave-one-out fit on its held-out dataset.

## 4. Verdict

- **Runs only if zrm's re-grade is ADOPT.** A gate (`zbase.py gate`) exits 0 only when both re-grades,
  `outputs/full_zrs/grade-nullx.json` and `outputs/full_zrm/grade-nullx.json`, are ADOPT. Otherwise the feeder drops
  the comparisons and grades behind it, and zrs is the base.
- **zrm becomes the base** if at least one primary read GAINs against zrs's fits and none of the 36 reads LOSEs.
- **Otherwise zrs stays the base.**
- **INCOMPLETE** (a comparison missing or not the declared one) decides nothing. zrs stays the base until a complete
  grade exists.

**What is already known, stated plainly.**
- zrs's 36 reads against zret's fits are filed.
- zrm's reads against zret's fits are known on five of six splits (its screen fits, L-2wiki, L-squad and L-metaqa).
- So this comparison is not blind. Its rule is the one every round uses, unchanged, and it is fixed before zrm's grade
  lands.

**Screens declared against zret's fits** keep their declared base: zgs (round thirteen) and zsep (round sixteen). If
one is ADOPTED, it is combined with the base in a later round, declared before its numbers.

## 5. Gates and order of work (8 October)

- **Code:** `outputs/mp_unified/zbase.py` (gate, base, compare, grade, regrade; relz.py's grade and nullx.py's re-grade,
  unchanged). It is committed with its selftest before any of its numbers.
- **`zb-base-S`** (zrs's fit against step 1's, six items) runs at once. These are zrs's R@5s, already filed against
  zret's fits.
- **`zb-gate`** waits for zrm's re-grade (`fzm-grade-nullx-b`).
- **`zb-compare-S`** (six items) waits on the gate, then **`zb-grade`**, then **`zb-grade-nullx`** (on the twelve null
  comparisons and the six `zb-base-S`).
- All CPU, 1 CPU each, 2 to 4 GB. They go ahead of zgs's and zsep's screen items in the feeder's list; nothing is
  preempted.
- **ETA:** the comparisons take a few minutes each. The verdict lands about 15 to 30 minutes after zrm's re-grade.

## 6. What this does not do

- No training, no read, and no new carve, seed, variant or hyperparameter. No new graph, text, encoder or model; the
  encoder and the substrate embeddings stay frozen.
- webqsp never trains. Test splits are never read.
- No speed figure. Any latency figure for the base is cold (8216ffe), timed in its own declared stage.

## Results
