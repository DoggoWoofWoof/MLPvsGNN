# PKO package on Kaggle: the served dataset package as private datasets (declared 30 Oct 2026)

**Status:** systems only. This file moves the served package to the user's own Kaggle account and verifies it. It computes no
number, runs no PKO stage, and edits no earlier file. A PKO stage that reads these datasets on Kaggle needs its own declaration
(a kernel, its memory, the carves it builds), as docs/PKOH_PKO_ON_THE_HOST.md and docs/PKOM_PKO_ON_MODAL.md do for the host and
Modal.

**The instruction.** On 30 Oct the user, while PKOM's three section-5 checks ran on Modal (docs/PKOM_PKO_ON_MODAL.md section 12),
said "you can parallely upload the package to kaggle". Earlier the same day: "run the three checks now, if kaggle has input
then do it"; Kaggle had no copy of the package (15 private datasets, about 10.7 GB, none of it the package), which is why those
checks ran on Modal.

**What goes.** The 577 files of `outputs/host_mirror_six/manifest.json` (69,297,874,669 bytes, 64.5 GiB; the package's
RECORD_SHA256 pin is that manifest's `freeze_RECORD_SHA256`), read in place from the package root, which is read only: nothing
under it is written, moved or re-hashed, and no link is made into it (the laptop has too little free disk for a staging copy of
the whole package, so a chunk at a time is copied and removed).

**How.** Nine private datasets on the user's account, named `mpr-pkg-<group>[-k]`: small (squad, musique and the four shared
files, 3.10 GiB), metaqa, webqsp, hotpotqa-1..3 and 2wiki-1..3, each at most 9 GiB (the biggest file is 4.95 GiB). Files are
staged flat as `NNNN<suffix>` (Kaggle's resume files hit Windows' path limit on long names) with a `manifest.json` per dataset
(flat name, package-relative path, size, sha256, dataset). Every staged copy is hashed and must equal the package manifest's
sha256 before upload; after upload Kaggle's own listing must show exactly the manifest's files at their sizes, or the chunk is
not counted as done; the staging folder is then removed. Two chunks run at a time; a chunk Kaggle already lists completely is
skipped, so a rerun resumes. Nothing is public, and no account name is written into any record or document.

**Space.** The nine datasets add 69.3 GB to the 10.7 GB already there: about 80 GB of Kaggle's private-dataset quota (about
100 GB; not re-read from Kaggle today). PKO's pools (88.1 GiB with the looks pruned, 142.3 GiB before) never go there. Deleting any
Kaggle dataset is the user's.

**What it does not do.** It does not make Kaggle a PKO route (no kernel is declared); it does not replace PKOM, whose package is
mpr-u2d's `/pkg`; it does not push anything to the lab host (that stays a separate declaration). Nothing here is a result.

**Closing note (31 October 2026, 01:23 IST).** All nine private datasets are complete and each was checked against the
manifest by `pkg_data.py` ("ready and complete", failed: none). The package is 577 files and 69.30 GB (64.54 GiB):

| dataset | files | bytes |
| --- | ---: | ---: |
| mpr-pkg-small | 62 | 3,329,675,940 |
| mpr-pkg-metaqa | 38 | 6,728,153,686 |
| mpr-pkg-webqsp | 121 | 9,168,830,059 |
| mpr-pkg-hotpotqa-1 | 89 | 9,584,907,972 |
| mpr-pkg-hotpotqa-2 | 65 | 7,326,353,706 |
| mpr-pkg-hotpotqa-3 | 13 | 5,372,244,649 |
| mpr-pkg-2wiki-1 | 89 | 9,570,106,934 |
| mpr-pkg-2wiki-2 | 82 | 9,630,194,244 |
| mpr-pkg-2wiki-3 | 18 | 8,587,407,479 |

- **Time.** It began at 16:16 on 30 October and finished at 01:23 on 31 October, 9 hours 7 minutes on the clock, of which
  about 3 hours 10 minutes was waiting after the second stop for the go-ahead to restart.
- **Redone.** The shell that ran the upload was stopped twice by the tool's memory-pressure guard, at about 19:45 and about
  21:35. mpr-pkg-2wiki-1 and mpr-pkg-2wiki-2 were uploaded again from the start after the first stop (they finished at
  21:20, 4,546 s and 4,538 s), and mpr-pkg-2wiki-3 after the second (it had been 14 minutes in; the second try took
  2,318 s). The six chunks done before the first stop were not touched; the script reported each as already complete.
- **Space.** 69.30 GB here, plus the 10.7 GB of earlier private datasets noted above, is about 80 GB. Kaggle's own page is the
  authority on what is left of the account's private quota.
- **The CLI's message.** After each successful create the Kaggle CLI printed "got an unexpected keyword argument 'type'".
  It is cosmetic; the check above is the script's own, against the manifest.
- **Not done.** No kernel reads these datasets yet; a PKO or refit stage that does needs its own declaration. The lab
  host's copy of the package stays absent, a separate declaration.
