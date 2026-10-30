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

**Closing note.** To be filled in when the nine datasets are complete: the time, any chunk that had to be redone, and the
final listing counts.
