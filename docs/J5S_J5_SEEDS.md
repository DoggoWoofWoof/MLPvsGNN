# J5S: seeds 1 and 2 of B1b's J5 fits (zrc and zsp)

Declared 11 Nov 2026, before any of its fits or numbers. Script: `outputs/mp_unified/j5s.py` (`rows`, `fork`, `report`,
`--selftest`).

## 1. Question

B1b's J5 rows, zrc (the MLP) and zsp (the MLP with one propagation step), are read on B1's three held-out settings
(HippoRAG 2's MuSiQue, 2Wiki and HotpotQA; 1,000 questions each) from one fit each, at seed 0. U1a and B1F read the same
two fits on the other link carves. B1M already gives the main pair three seeds; these are the last one-seed rows of the
set. This stage asks two things:
- How far do these rows move with the training seed?
- Does any of their calls change when each question's R@5 is averaged over seeds 0, 1 and 2?

## 2. What runs

- **Fits.** zrc and zsp on J5 at seeds 1 and 2. Each uses the filed row's own command (`fzk-train-J5`, `fsp-train-J5`)
  with only three changes: `--name J5-s<k>`, `--seed <k>` and `--out-root outputs/j5s/train/<arm>`. Every other setting
  is the filed fit's: split J5; train metaqa=fit, squad=fit, musique=s1fit, hotpotqa=fit, 2wiki=fit; basis 2wiki;
  variant p; config 2e-3:1e-4:0.1:8:2; hidden 128. `fork` refuses a fit whose screen.json differs in any of these.
  `--seed` reaches lean_gpu's `fit_variant`, which seeds torch, numpy's generator (batch order) and the model's own
  generators.
- **Control fits.** One seed-0 fit per arm (`J5-s0c`): the same command at `--seed 0`.
- **Forks.** Each fit is copied into `outputs/j5s/fits/<carve>/<arm>/J5-s<k>` for the four carves, without its reads.
  - Seed 0's copy is the filed fit itself (`outputs/full_zrct/fits/J5`, `outputs/full_zsp/fits/J5`), checked against
    its filed sha256 (zrc 38ad128d..., zsp 966796ea...).
  - The control goes into b1 only.
- **Reads.** Each copy is read on its carve with the commands of the filed read rows (`b1-read2-zrc-J5`,
  `b1-read2-zsp-J5`), changing only the name, the output folder and the carve. The carves:
  - b1: the corpus hyperlinks (MuSiQue: its title links), as B1b;
  - b1w: W2_S's rule links, as B1F;
  - b1t: text links only, as U1a;
  - b1u: U1a's title rule, as U1a.
- **Count.** 26 reads: 2 arms x 3 seeds x 4 carves, plus the two controls on b1.

## 3. Inputs checked before declaring (11 Nov, on the host)

- Every script the fits and reads load has its recorded sha256; lean_mlp5 differs from its record by line endings
  only. torch 2.8.0 and numpy 2.3.2 are as recorded.
- Every training record the two filed fits list is byte-equal to the host's current file: the step1 parts and the
  basis, and the rmatch, zrc and zlink records of the train carves.
- Every read carve's step1 part has the sha256 that each filed read.json records, and every filed read file has its
  recorded sha256.
- No step1 or zlink record of a read carve was written after the read that uses it. MuSiQue's b1 zlink record was
  written between the filed zrc read and the zsp read; only zsp reads zlink.

## 4. Controls

These are filed whatever they show.
- **C1.** Each control fit against the filed fit, tensor by tensor over all nine candidates' states, plus every other
  field of models.pt except the fit's name. lean_gpu saves the name in the file, so the file sha256 always differs.
  The call is REPRODUCED or NOT_REPRODUCED, with the two loss curves beside it.
- **C2.** The seed-0 re-reads against the filed reads, array by array (12 per arm). Each is EQUAL, DIFFERENT (with
  the R@5 change) or MISSING.
- **C3.** The controls' b1 reads against the filed b1 reads, the same way.

If C1, C2 or C3 fails, the report says so, and the record names the difference beside the filed numbers. The seed-0
numbers in the three-seed means are this stage's re-reads. Nothing is rerun to make a control pass.

## 5. Report (`outputs/j5s/report.json`, `.md`)

- **Per setting, carve and arm:**
  - each seed's R@5;
  - the three-seed mean, with each question's R@5 averaged over the three seeds (B1M's rule), and B1b's 95% bootstrap
    interval over the questions (2,000 resamples, seed 0);
  - the seed spread;
  - the mean FC@5 and hit@1;
  - RRF's R@5, which must be equal across the seeds (checked; the report stops otherwise).
- **Calls.** B1b's rule: ABOVE when the interval's low end is above the reference, BELOW when its high end is below,
  AT otherwise. Calls are made against HippoRAG 2 and against each row of B1F's published list, from the three-seed
  interval and from seed 0's alone. Every call that changes is listed with its direction.
- **Paired differences on the three-seed means:**
  - b1-b1t, b1w-b1t, b1u-b1t, b1-b1w and b1w-b1u, per arm;
  - zsp-zrc, per carve;
  - B1F's share of the hyperlink gain: (b1w-b1t)/(b1-b1t) and (b1u-b1t)/(b1-b1t);
  - (zrc-RRF)/(zsp-RRF) on b1.

## 6. Record rule

From the report on, a J5 row on these settings is quoted as its three-seed mean with its interval, with seed 0's value
beside it. Any call that changes is named with its direction. A call that holds at seed 0 but not at three seeds is no
longer quoted as holding. This stage decides nothing about models: it adopts or drops no fit.

## 7. Resources and timing

- **zrc fits:** on the card at the filed cap (0.26, share 0.28), three at a time; 4 GB reserved each (2.4 GB
  measured).
- **zsp fits:** on the CPU, 6 threads, 12 GB each (9.0 GB measured).
- **Reads:** zrc 3 GB with card share 0.32 (cap 0.3, as filed); zsp 6 threads and 6 GB.
- **Measured on 8-9 Oct:** a zrc fit took 72 min and a zsp fit 82 min; a read takes 13-94 s.
- **Order:** the GPU fits are queued first.

## 8. Files

- `outputs/j5s/train/<arm>/J5-s<k>/`: models.pt, train.json, screen.json.
- `outputs/j5s/fits/<carve>/<arm>/J5-s<k>/`: fork.json and reads/.
- `outputs/j5s/report.json` and `report.md`.
- Feeder rows `j5s-*`. A `j5s-selftest` row gates the block; it checks the real queue's template rows, the filed reads
  and the filed fits' sha256.

## 9. Outcome, 11 Nov: stopped at seed 0's b1w reads

- **What ran.** The 42 rows were queued at 20:51. The self-test passed on the host, and both seed-0 forks copied the
  filed fits with their filed sha256. The seed-0 reads on b1w ran for both arms.
- **C2 on b1w.** All six seed-0 re-reads equal the filed B1F reads: the same 11 arrays each, with the same dtype, shape
  and values (`outputs/j5s/c2_b1w.json`). The file sha256s differ only through the time stamps inside the zip files.
- **What stopped.** All six fits (seeds 1 and 2, and both controls) stopped after 3-7 s with FileNotFoundError on
  `outputs/step1/cache/metaqa/fit/part_0of8/xc.npy`. The seed-0 reads on b1, b1t and b1u stopped the same way after
  2 s. C1, C3 and the other six C2 comparisons were not reached.
- **Cause.** The host's auto-cleanup removes cache arrays it can rebuild and keeps each part's record.json. Section 3
  compared records and read files, not the arrays the records list. A read-only count on the host after the failures
  found these arrays missing:

  | Used by | Cache | Arrays missing | GB |
  |---|---|---|---|
  | fits | step1 | 252 of 289 | 6.19 |
  | fits | rmatch | 32 of 56 (metaqa/fit) | 0.50 |
  | fits | zrc | 32 of 56 (metaqa/fit) | 0.50 |
  | fits | zlink | 51 of 68 | 1.97 |
  | reads | step1 | 93 of 204 (b1, b1t, b1u; b1w complete) | 1.06 |
  | reads | zlink | 9 of 48 (MuSiQue b1, b1t, b1u) | 0.38 |

  That is about 10.6 GB in all.
- **Not rebuilt.** A rebuild would take the host's free disk from 39.3 GB to below its 30 GB floor, and would add
  hours of builds whose arrays have not yet been checked against the records. The other 37 rows are held behind
  `hold-j5s-caches`.
- **Record.** No three-seed number exists. Every J5 row stays quoted from its one seed-0 fit, as before this stage.
- **To resume.**
  1. Copy every record.json first.
  2. Rebuild the missing arrays with each record's settings (blas_threads 2).
  3. Check every rebuilt array's sha256 against the copied record, and refuse on any mismatch.
  4. Release the hold.

## 10. Amendment A1, 11 Nov 23:35 (declared before any look, cache or fit runs): restoring the removed looks and caches

Systems only. No fit, read, control, report or record rule of sections 2-6 changes, and this section holds no result.

- **What section 9 missed.** The cache arrays are built from the look chunks
  (`outputs/mp_unified/look/<dataset>/<carve>/chunks/`), and the auto-cleanup removed those chunks as well, again keeping
  every record. Eight looks lost chunks, about 1,200 in all: the fit carves of metaqa, squad, musique (s1fit), hotpotqa
  and 2wiki, and MuSiQue's b1, b1t and b1u. Rebuilding the arrays alone cannot work. The other carves J5S loads (b1w,
  and the b1, b1t and b1u of 2wiki and hotpotqa) still hold their chunks.
- **Rule.** Each file is rebuilt by its own committed script with its original command and settings, and goes back only
  where its sha256 is the one a record lists. A file that exists is never touched. Section 9's four steps hold: step 1 is
  `backup`, step 2 the rows below, step 3 the look-checks and installs, step 4 the verify gate.
- **Script.** `outputs/mp_unified/j5s_restore.py`, committed with this section; its self-test passes on toy trees and it
  has read no record. It imports the helpers of N2's restore (`scripts/p1c2_cache_restore.py`, not edited).
  1. `backup`: before anything runs, a copy of every look json that a look rewrites (record*.json, b1*.json, ids.json,
     info.json of the eight looks) and of every part's record.json in the four caches, for the carves J5S loads, in
     `outputs/j5s/restore/backup/`, with their sha256s in backup.json. A backup that exists must hold the same bytes.
  2. The eight looks run again with their original commands, in place (24 shards: musique s1fit 6, metaqa 6, and 2 each
     for the others). A look skips every chunk that exists and writes the missing ones, and rewrites its shard's records.
  3. `look-check`, once a carve's shards are done: each rewritten json against its backup. ids.json and info.json byte
     for byte; each record on every key, at any depth, except seconds, utc, peak_rss_bytes and placement. A code sha that
     differs (a look script, the copied look_score_kb, B1a's script, u1a's script) is reported, not refused. Every chunk
     the records list must be on disk. If all of that holds, the backup's bytes go back over the rewritten files, so
     every cache record that names a look record's sha256 holds as filed. Anything else refuses (exit 2) and nothing is
     put back.
  4. `cache`: step 1's arrays by `lean_cache.py` (two BLAS threads, as the records), each part with its own split, bases
     and the default look root, into `outputs/j5s/rebuild/step1`; a carve's parts run side by side, one row each, and the
     carve's own row builds any part still missing and gives the carve's verdict. `chains`: rmatch and zrc (metaqa/fit)
     and zlink (eight carves) by the call each CLI makes for `build --host`, with only out_root changed to the rebuild
     root, after the carve's look is checked and its step-1 cache is whole (zrc also waits for rmatch).
  5. `install`: for each part, the rebuilt record must equal the original on every key except seconds, utc, placement,
     peak_rss_bytes, row_gather_s and the arrays' sha256s; a code sha that differs is reported, not refused. Each missing
     array goes back, as a hard link or else a copy renamed into place, where the rebuilt file's sha256 is the one the
     record lists. Then the carve is hashed against its records.
  6. `verify`: every array of every carve J5S loads, hashed against its record; step 1's two bases; the eight
     look-checks. WHOLE or NOT_WHOLE; exit 1 unless WHOLE.
- **What is checked, and what is not.** The look records list no chunk sha256s, so a rebuilt chunk is checked only through
  the arrays built from it, each against its record. An array that does not reproduce is never installed. It stops its
  carve, `verify` stays NOT_WHOLE, and no J5S row runs. A changed code sha is not a reason to refuse by itself; the
  arrays decide.
- **Rows.** 72, named `j5r-*`: `j5r-backup`; `j5r-look-<carve>-<i>` (24); `j5r-lookchk-<carve>` (8);
  `j5r-cache-<carve>`, with `-p<i>` rows for metaqa/fit (8 parts) and musique/s1fit (6) (28); `j5r-rm-metaqa_fit`,
  `j5r-zrc-metaqa_fit`; `j5r-zl-<carve>` (8); `j5r-verify`. Resources: looks 1.4 CPUs as their original rows, memory
  from the records' peaks plus about 0.5 GB (6.7-8.6 GB); caches 1.5 CPUs and 1.4-4.8 GB; chains 1 CPU and 2-4 GB as
  their original rows. Commands, inputs and resources are in the script (`python outputs/mp_unified/j5s_restore.py rows`).
- **J5S rows.** The 37 held rows are queued again under new names, `j5s2-*` (the feeder never sends a name twice). Each
  command, input, resource and order is the declared row's; the only changes are the name and the hold: `j5r-verify` in
  place of `hold-j5s-caches`. The five rows that ran (`j5s-selftest`, the two seed-0 forks and the two b1w reads) keep
  their names, and the rows that depend on them keep those names.
- **Keep row.** `keep-j5s-11nov`, held, names the eight look folders, the step-1 and zlink folders of the 17 carves,
  `outputs/rmatch/cache/metaqa/fit/`, `outputs/zrc/cache/metaqa/fit/` and `outputs/j5s/`, so that the auto-cleanup cannot
  remove them again before the J5S rows finish. It is released when J5S is filed.
- **Disk and time.** About 23 GB of new files by estimate against about 81 GB free at the last check (floor 30 GB); the
  free disk is read on the host again before release. The longest look shard in the records is 3,938 s (musique s1fit),
  a step-1 part takes about 8 minutes, zlink about 3. With the J5S fits (72-82 min) and reads after them, about 4 hours.
- **Files.** `outputs/j5s/restore/` (backup/, backup.json, `look_<carve>.json`, `install_<kind>_<carve>[_p<i>].json`,
  verify.json) and `outputs/j5s/rebuild/` (the side roots).
- **Record.** If every step passes, J5S runs as declared in sections 2-6 and its report is filed as sections 5 and 6
  say. If any step refuses, the stage stops there and the refusal is filed in a section 11; no three-seed number is
  quoted from a partly restored tree.
