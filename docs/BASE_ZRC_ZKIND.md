# The next base now that zrc and zkind are both ADOPTED: zkind's fits against zrc's, read by read

Declared 9 October 2026 at about 08:00. This is after zrc's re-grade (ADOPT, docs/FULL_ROUND18.md, filed 03:35) and
zkind's (ADOPT, docs/FULL_ROUND19.md, filed 07:50), and before any comparison of the two. docs/FULL_ROUND19.md, section
4: "If round eighteen's zrc is ADOPTED against zrm as well, the base is decided in a file declared before that decision,
as docs/BASE_ZRM_ZRS.md decided between zrm and zrs." This is that file. Numbers here are development numbers; the
paper's numbers come from one declared confirmation run.

## 1. Question

zrc (round eighteen) is zrm with each typed row keeping the 64 entries w0 ranks highest. zkind (round nineteen) is zrm
plus a head over the typed graph's kinds. Both are ADOPTED against zrm's fits. Which is the base of every later screen and
run?

## 2. The incumbent is zrc

- zrc was ADOPTED first.
- It changes less. It changes only which entries a typed row keeps (preprocessing); the model is zrm's. zkind adds a
  learned head to the model.
- So zkind must beat zrc with nothing lost, read by read, to become the base.

## 3. What is compared

- Nothing trains or reads, and no fit folder is written.
- **zkind's six fits against zrc's,** split by split, p@swa on the six s1eval carves: 36 reads.
  - zkind's fits: `scr-zkind` and `scr-zkind-hp` (its screen fits) on L-musique and L-hotpotqa, and
    `outputs/full_zkind/fits/S` on the other four splits.
  - zrc's fits: `scr-zrct` and `scr-zrct-hp`, and `outputs/full_zrct/fits/S`.
  - zrm's and step 1's fits of each split are reported beside these and decide nothing.
- The comparison is lean_screen's: the R@5 difference with a 2,000-resample question bootstrap, filed at the floor
  0.0075 (`outputs/zbase2/compare-S`). The grade is relz.py's, decided against zrc's fit of each split
  (`outputs/zbase2/grade`).
- **The re-grade decides.** `nullx.py regrade` calls every read under the seed null over every split
  (`outputs/zbase2/grade-nullx`). Each read's base R@5 is zrc's: zrc's six fits compared with step 1's
  (`outputs/zbase2/base-zrc-S`).
- **Primary reads** are step 1's eleven: J5 on all six datasets, and each leave-one-out fit on its held-out dataset.

## 4. Verdict

- **The gate** (`zbase2.py gate`) exits 0 only when both re-grades, `outputs/full_zrct/grade-nullx.json` and
  `outputs/full_zkind/grade-nullx.json`, are ADOPT. Both are.
- **zkind becomes the base** if at least one primary read GAINs against zrc's fits and none of the 36 reads LOSEs.
- **Otherwise zrc is the base.**
- **INCOMPLETE** (a comparison missing or not the declared one) decides nothing. zrm stays the base until a complete
  grade exists.

**What is already known, stated plainly.**
- Both arms' 36 reads against zrm's fits are filed. Both gain on webqsp read zero-shot (zrc +0.026 to +0.051 on four
  fits, zkind +0.017 to +0.059), and both read +0.0000 on all six datasets in L-metaqa's fit.
- So this comparison is not blind. Its rule is the one every round uses, unchanged, and it is fixed before any read of
  zkind against zrc exists.

**Their combination is a round of its own.** zrc changes which entries a typed row keeps, and zkind the head that reads
them, so the two do not overlap. Whichever is the base, zrc's entries under zkind's head is declared in a later round,
before its numbers, and decided against the base.

**Screens declared against zrm's fits** keep their declared base: rounds twenty-one (zrg), twenty-two (zlk, its full run
flk), twenty-six (zgf) and twenty-seven (zgr). If one is ADOPTED, it is combined with the base in a later round, declared
before its numbers.

## 5. Gates and order of work (9 October)

- **Code:** `outputs/mp_unified/zbase2.py` (gate, base, compare, grade, regrade). It is zbase.py's code for this pair:
  zbase.py is frozen, and relz.py's grade and nullx.py's re-grade are unchanged. It is committed with its selftest
  before any of its numbers.
- **`zb2-base-S`** (zrc's fit against step 1's, six items) and **`zb2-gate`** run at once. zrc's R@5s are already
  filed against zrm's fits.
- **`zb2-compare-S`** (six items) waits on the gate, then **`zb2-grade`**, then **`zb2-grade-nullx`** (on the twelve
  null comparisons and the six `zb2-base-S`).
- All CPU, 1 CPU each, 2 to 4 GB, at the top of the feeder's list; nothing is preempted.
- **ETA:** the comparisons take a few minutes each. The verdict lands about 08:30.

## 6. What this does not do

- No training, no read, and no new carve, seed, variant or hyperparameter. No new graph, text, encoder or model; the
  encoder and the substrate embeddings stay frozen.
- webqsp never trains. Test splits are never read.
- No speed figure. Any latency figure for the base is cold (8216ffe), timed in its own declared stage.

## Results
