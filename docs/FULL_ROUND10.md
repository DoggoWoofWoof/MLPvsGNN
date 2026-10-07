# Full run: ztop50 on rel (relz) on all six splits, only if its screen's re-call is PROMISING and rel is ADOPTED

Declared 8 October 2026 at about 02:35, before any number of round ten's screen exists (docs/SCREENS.md, tenth round).
The run starts only if two things hold:
- relz's pair, re-called under the seed null (`outputs/screen/scr-relz-pair-recall`), is PROMISING;
- rel's full run, re-graded (`outputs/full_rel/grade-recall`, docs/FULL_ROUND9.md), is ADOPT.

Otherwise none of it happens, and this file records why. Numbers here are development numbers; the paper's numbers
come from one declared confirmation run.

## 1. Question

relz puts ztop50's pool normalisation on top of rel's inputs. Every block's within-pool z-score, rel's included, and
rrf's base z-score are taken against the pool's top 50 retrieved rows by rrf, not the whole pool.

If the screen's re-call is PROMISING: does its gain over rel hold on every split, and does nothing else lose? The
splits are J5 (in-domain on all five training datasets) and each leave-one-out fit on its held-out dataset. The screen
reads musique, hotpotqa and webqsp zero-shot. Only this run reads metaqa, squad and 2wiki zero-shot, and every dataset
in-domain under J5.

## 2. Fits

- **Splits:** step 1's six seed-0 splits, with step 1's fit carves and basis.
- **The screen's fits are this run's L-musique and L-hotpotqa fits** (`outputs/screen/fits/scr-relz` and
  `scr-relz-hp`). The run trains four fits, in the order L-2wiki, L-squad, J5, L-metaqa, through
  `outputs/mp_unified/relz.py train --split S --name S --arm relz --out-root outputs/full_relz/fits`.
- **One fit per split:** variant p, seed 0, config 2e-3:1e-4:0.1:8:2, hidden 128, the SWA state p@swa. No select
  carves, no pick.
- **relz on L-metaqa is ztop50 on step 1's blocks.** That split trains no KB, so rel's blocks are 0 on every training
  row and lean_gpu's dead-block rule drops them; rel's own L-metaqa fit is step 1's model. This fit then compares
  ztop50 with step 1 on L-metaqa, a split no ztop50 fit has read.
- **On the card,** rel's caps: J5, L-squad and L-2wiki 0.30 (share 0.32), L-metaqa 0.22 (share 0.24), reads 0.28
  (share 0.30). ztop50 adds a sort per batch and no columns.

## 3. Reads and comparison

- Each fit is read on the six s1eval carves.
- Each read is compared, question by question, with rel's fit of the same split, p@swa, on the same carves:
  `outputs/full_rel/fits/S`, and rel's screen fits for L-musique and L-hotpotqa. That comparison decides. Step 1's fit
  of the split is reported beside it.
  - The comparison is lean_screen's: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS or
    WITHIN with the floor 0.0075.
  - Twenty-four reads come from these fits and twelve from the screen: 36 in all.
- **Primary reads** are step 1's eleven: J5 on all six datasets (five in-domain, webqsp zero-shot), and each
  leave-out fit on its held-out dataset.

## 4. Verdict

- `relz.py grade --full-root outputs/full_relz --reuse L-musique=outputs/screen/scr-relz.json,L-hotpotqa=outputs/screen/scr-relz-hp.json`
  files the grade (lean_screen2's rule, decided against rel's fits; a comparison decided against any other base makes
  it INCOMPLETE).
- `relz.py regrade` re-calls its twelve reused screen reads under the seed null's floors, with rel's screen R@5 as
  each read's base, into `outputs/full_relz/grade-recall.{md,json}` (filed for the record).
- **Amended at about 02:45, before any of relz's numbers:** `nullx.py regrade --base-compares` re-grades all 36 reads
  under the seed null over every split, with rel's R@5 as each read's base (rel's six comparisons with step 1's fits),
  into `outputs/full_relz/grade-nullx.{md,json}`. **That re-grade decides.**
- **ADOPT:** at least one primary read GAINs, and none of the 36 reads LOSEs.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a comparison is missing or is not the declared one (exit 1).

**After ADOPT,** relz becomes the base of every later screen and run, in rel's place.

**The first LOSS settles NOT_ADOPTED.** The fits still running are then stopped, and the ones not yet started are
taken out of the queue, since no read they could give changes the verdict.

## 5. Gates and order of work (8 October)

- **`frz-gate`** (`relz.py gate --recall outputs/screen/scr-relz-pair-recall.json --rel-grade
  outputs/full_rel/grade-nullx.json`) exits 0 only when the re-call is PROMISING and rel's re-grade is ADOPT.
  - Amended at about 02:45, before any of relz's numbers: rel's re-grade is the one under the seed null over every
    split (docs/SCREENS.md, section 2), filed in place of `grade-recall`.
  - The four fits wait on it. On any other result the feeder drops them with their reads, comparisons and grades.
- **If rel is NOT_ADOPTED,** relz's screen is still filed. It says whether ztop50 adds to rel, but there is no adopted
  rel to build on. A run against step 1 would need its own declaration.
- **The items** go after gsurg's `-rc` run in the feeder, so on the card they start after its fits.
- **ETAs if it passes:** rel's re-grade lands about 03:15 to 03:45 and relz's re-call about 04:15 to 05:00. The fits
  take about 45 to 60 minutes each beside the others. The grade comes about 06:30 to 07:30, sooner at a LOSS.

## 6. What this does not do

- No new carves, seeds, variants, columns or hyperparameters. relz is two filed parts (relcols.py's rel, lean_screen4's
  ztop50) used unchanged.
- No new graph, text, encoder or model. The encoder and the substrate embeddings stay frozen.
- webqsp never trains. Test splits are never read.

## Results
