# B1 re-call: every model decision, called again without B1's questions

Declared 15 October 2026, before any re-called number exists. Stage 9 of docs/PROGRAM_2026_10_15.md. Code:
`scripts/b1_recall.py`. Status: **STANDS** (section 6).

## 1. Why

HippoRAG 2's benchmark (B1, Claim 2) holds 1,000 questions each of musique, hotpotqa and 2wiki. They are rows of the
dev splits that s1eval holds whole: 1,000 of musique's 2,417, 1,000 of hotpotqa's 7,405 and 1,000 of 2wiki's 12,576.
No fit trained on them, and nothing was chosen on the B1 runs themselves. But every screen and full run since 7
October read them when it decided (PROJECT_STORY section 4.8, audit MEDIUM). The check proposed on 14 October: call
every adoption decision again without B1's questions. If every call stands, our choices did not depend on them.

## 2. Inputs (read only)

- B1's question ids: `outputs/bench/hipporag2/{musique,hotpotqa,2wiki}/queries.json` (`id`, the package's query id).
- Every filed decision record:
  - the screens' re-calls under the seed null, `outputs/screen/*-pair-recall.json`. When a pair's own record is not on
    the host, its two comparisons are found by the screens' names (`NAME.json`, `NAME-hp.json`). The replay must still
    give every filed row;
  - the full runs' re-grades under the seed null over every split, `outputs/full_*/grade-nullx.json` and
    `outputs/zbase2/grade-nullx.json`.
- Through them, the records each one names: the pair or the grade, the comparisons (`compare` records, each naming
  the fit read and the fit it is decided against), and the null's comparisons (seeds 1 and 2 of step 1's base arm
  against seed 0).
- The fits' filed s1eval reads (`reads/{ds}__s1eval.npz`), on the host. No model is run. A fit's read is used exactly
  as filed, with B1's questions dropped or kept.

## 3. The replay

For each decision, every read (split, dataset) is computed again from the two fits' filed reads, as lean_screen's
`compare` computed it:
- per question R@5, FC@5 and hit@1 of candidate p@swa, questions with golds only;
- the difference's mean and its 2,000-draw question bootstrap (seed = the dataset's place in EVAL_ORDER);
- the call at 0.0075.

The null's seed-only differences D1 and D2 are recomputed the same way, and so is each read's floor,
max(0.0075, 2·sqrt((D1² + D2²)/2)). The re-call under the floor and the verdict follow screen_recall.py and nullx.py,
imported unchanged:
- a screen pair is PROMISING, MIXED or NO_GAIN;
- a full run is ADOPT when a primary read GAINs and none of the 36 LOSEs.

Every number is rounded where the filed chain rounds it (4 places), so the replay with every question must give the
filed record.

**Two passes:**
1. **All questions (the check).** The replay must reproduce each decision's filed record: every read's difference and
   interval, every null difference, every call and the verdict. A decision that does not reproduce is reported
   `NOT_REPRODUCED` and is not re-called. A decision whose reads are no longer on the host is `NOT_REPLAYABLE`.
2. **Without B1.** The same replay with B1's 3,000 questions dropped from musique's, hotpotqa's and 2wiki's reads
   (both fits of every comparison, and the null's). Nothing else changes.

## 4. What it decides

**The lineage** is the decisions behind the models we report (17 records, named in the script's `LINEAGE`):
- zret's full run; zrs's screen and full run; zrm's screen and full run, and zrm over zrs (`zbase`): the MLP base;
- zrc's screen and full run (round 18's zrct);
- zrc as the base over zkind (`zbase2`);
- zsp's screen and full run (the GNN track);
- P1c3's screens and full runs (zrck, zspk);
- P1c3p's screen and full run (zrckp).

**The verdicts:**
- **`STANDS`:** every lineage decision that reproduces keeps its filed verdict without B1.
- **`CHANGED`:** at least one does not. The changed decision is named, and so is what it means: a model built on it
  was chosen in part on B1's questions. The B1 numbers then carry that decision's caveat by name. Nothing is
  re-adopted or withdrawn by this file. `CHANGED` holds even when another lineage decision could not be checked.
- **`NOT_ALL_CHECKED`:** none changes, but a lineage decision was `NOT_REPRODUCED` or `NOT_REPLAYABLE`. Each one is
  named, and STANDS is not claimed for it.

Every other filed decision is re-called beside, with its verdict and each changed call. A decision outside the
lineage cannot change the verdict.

Pool decisions are not in this file. Step 4b's own-graph walk and step 4e's A3 links were chosen reading s1eval's pool
coverage. P1e chose KIND_T on s1sel (B1-free) and called it on s1eval. P1f chooses its size rule on s1sel only. The two
pool choices made on s1eval are listed as open in section 6.

## 5. Running it

- **Selftest** (`python scripts/b1_recall.py --selftest`, passed before this commit):
  - the filter on a hand-built read (every array cut along its question axis, the candidates kept);
  - the floors and calls against screen_recall's on hand-built numbers;
  - the replay of a fake decision written with lean_screen's own compare, reproduced exactly;
  - a fake decision with a changed number reported NOT_REPRODUCED;
  - one whose records are gone reported NOT_REPLAYABLE.
- **Host item** `b1r-recall`: one CPU item (the reads are on the host), about 30 minutes.

## 6. Results

**`STANDS`** (`outputs/b1r/b1r.{json,md}`, filed 15 October about 15:47; host item `b1r-recall`, 15 minutes).

- **Every decision reproduced.** All 60 filed decisions (17 lineage, 43 beside) give their filed record exactly when
  replayed with every question: 1,176 reads, every difference, interval, floor, call and verdict. None was
  NOT_REPRODUCED or NOT_REPLAYABLE, and no pair needed the fallback by name.
- **No lineage verdict changes without B1.** All 17 keep their filed verdict, so the models we report (zrckp, zspk and
  the bases under them) were not chosen because of B1's questions.
- **Six lineage reads change their call, all on musique, all GAIN to WITHIN.** These are in scr-p1c3-zspk,
  scr-p1c3p-zrckp, full_zrckp (two), full_zret and full_zspk. For example, full_zrckp's L-hotpotqa read of musique goes
  from +0.0100 to +0.0080. B1 holds 1,000 of musique's 2,417 s1eval questions, so without them the intervals widen and
  these small gains no longer clear zero. Each of those decisions keeps another GAIN, so no verdict moves.
- **Beside, two screen verdicts change** (they cannot change this file's verdict): scr-zgs PROMISING to MIXED (its
  L-hotpotqa musique read goes from WITHIN to LOSS, −0.0065 to −0.0096) and scr-zlk PROMISING to NO_GAIN (musique GAIN
  to WITHIN). Neither is behind a reported model.

**Still open:** the two pool choices made reading s1eval's pool coverage (step 4b's own-graph walk, step 4e's A3
links) are outside this file. B1's questions should also stay out of the confirmation run's held-out questions.
