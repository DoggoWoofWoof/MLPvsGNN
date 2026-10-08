# Full run: zfs, the selected feature blocks on zrm, only if its screen's re-call is PROMISING and zrm is the base

Declared 8 October 2026 at about 20:45, before any number of round twenty-four exists (docs/SCREENS.md, twenty-fourth
round). zrm is the base (docs/BASE_ZRM_ZRS.md, re-graded ADOPT at 10:05). These are development numbers. The paper's
numbers come from one declared confirmation run.

**No message passing.** Every input is a look column computed before any row is scored (docs/SCREENS.md, twenty-fourth
round).

## 1. Question

Does adding the subset of the seven candidate blocks that the screen selected (structure in the NER, kNN and
all-family views, all-family depth, seed similarity, relation text) to zrm lift its reads on every split? And does
nothing else lose? The splits are J5 (in-domain on all five training datasets) and each leave-one-out fit on its
held-out dataset.

## 2. Fits

- **What trains.** The arm zfs (zfeat.ZFS over zfeat.FeatChainCarve), as in the screen.
  - It uses zrm's model, settings and training, with the seven blocks after the pick's.
  - The candidates are dropped at 0.5 per (question, block) in training.
  - Seed 0, on the card.
- **The subset is the screen's.** It comes from `outputs/zfeat/select.json` and is not selected again. Every read keeps
  that subset and masks the other candidates.
- **Fits.** Four: L-2wiki, L-squad, L-metaqa and J5.
  - Command: `zfeat.py train --split S --name S --arm zfs --out-root outputs/full_zfs/fits --device cuda`.
  - The screen's two fits (scr-zfs, scr-zfs-hp) are this run's L-musique and L-hotpotqa fits.
- **Builds.** The candidates' columns of every carve these fits read, built as the screen's.

## 3. Reads and comparison

- Each fit is read on the six s1eval carves with the subset kept.
- **Each read is compared, question by question, with zrm's fit of the same split.** That is zrm's full run
  (`outputs/full_zrm/fits/S`) or its screen fits (scr-zrm, scr-zrm-hp), with step 1's beside.
  - The comparison is lean_screen's: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS or
    WITHIN with the floor 0.0075.
  - Twenty-four reads come from the four fits and twelve from the screen: 36 in all.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot) and each leave-out
  fit on its held-out dataset.

## 4. Verdict

- `zfeat.py grade --full-root outputs/full_zfs --reuse
  L-musique=outputs/screen/scr-zfs.json,L-hotpotqa=outputs/screen/scr-zfs-hp.json` files the grade: relz.py's grade,
  decided against zrm's fits, under zfs's name.
- **The re-grade decides.** `nullx.py regrade` re-grades all 36 reads under the seed null over every split into
  `outputs/full_zfs/grade-nullx.{md,json}`. Each read's base R@5 comes from zrm's (`outputs/zrc/base-zrm-<split>.json`).
- **ADOPT:** at least one primary read GAINs, and none of the 36 LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing (exit 1).

**ADOPT** makes zfs, with its subset, the MLP's base for every later screen. If another round is also ADOPTED against
zrm, the base is decided in a file declared before that decision, as docs/BASE_ZRM_ZRS.md decided between zrm and zrs.

**Every fit runs to its end.** A LOSS at the floor may turn WITHIN under the null.

## 5. Gates and order

- **`ffs-gate`** (`zfeat.py gate`) exits 0 only when both hold:
  - the screen's re-call (`outputs/screen/scr-zfs-pair-recall.json`) is PROMISING;
  - zrm's re-grade (`outputs/zbase/grade-nullx.json`) is ADOPT.
- The full run's items are queued after the screen's re-call, behind the gate, as card items after the screens waiting
  then.

## 6. What this does not do

- No new graph, column, text, encoder or embedding. No new seed or hyperparameter beyond the screen's 0.5 dropout on
  the candidates.
- No selection on an evaluation carve.
- Nothing is written into zrm's, zret's or step 1's fit folders.
- webqsp never trains. Test splits are never read.
- **Speed.** The candidates are compiled with the pick's per question. Any latency figure is cold (8216ffe) and is
  declared in its own stage.

## Results
