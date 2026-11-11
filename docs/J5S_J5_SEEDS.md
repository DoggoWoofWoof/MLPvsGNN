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
