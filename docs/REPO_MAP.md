# Repository map: where things live, what is canonical, and what is a candidate to remove

Written 28 Nov 2026 (ING1 amendment 2 and the 28 Nov clean-up request). It is a map, not a rule: the rules stay on each stage's own page. Every number below was read from the repository on 28 Nov.

## 1. The canonical entry points (one place for each question)

| Question | The one place | Checked by |
|---|---|---|
| Which datasets exist, where is each package, which encoder arms | `configs/datasets.json` (registry; the `encoder_policy` key names the defined encoder) | `python -E -P -B scripts/ingest/ingest.py check` |
| Which rule set builds the links and the pools | `configs/rule_sets.json` | `ingest.py plan` |
| How a package is read (nodes, queries, golds, vectors) | `outputs/mp_unified/zworld.py` (edit only through an amendment on `docs/ING1_INGESTION_AND_STAGE_REGISTRY.md`) | its gate G-W |
| Which encoder a module may read | `encoder_policy` in the registry and `configs/legacy_encoder_modules.json` (modules that read the older encoder's lists, each with its reason) | `scripts/ingest/encoder_policy.py check`, `tests/test_encoder_policy.py` |
| What runs on the host and in what order | `outputs/host_ops/hostfeed_items.txt` (one pipe-separated row per job: line, x, deps, name, cpus, mem, inputs, command, optional gpu field) | `hostfeed2.parse_items`, `feed_backlog.py` |
| What a stage declared and found | `docs/<STAGE>_<TITLE>.md`, one page per stage, amendments appended as numbered sections | the stage's own gates |
| Where the state of the work is summarised | `docs/CHECKPOINT_*.md` (latest: `docs/CHECKPOINT_2026_11_26_PER_DATASET.md`) | its sections' dates |

## 2. Names

- **Stage code** is `outputs/mp_unified/z<stage><n>.py` (for example `zjnt2.py`, `zstk3.py`, `zdrp1.py`); a module is declared on its stage page before a row runs it and is frozen once a host row has run it, so a change is an amendment plus a new row name, not an edit.
- **Stage records** are `outputs/<stage>/` (for the amended JNT1 step `outputs/jnt2/<dataset>/`); a record is committed with `git add -f`, an array (`.npz`, `.npy`) never is.
- **Rows** on the host are `<stage>-<tag>`; the feeder's class is the first two name tokens and must be unique among new rows.
- **Link labels** are S (structure: hyperlinks or KB triples), R (rule links; components W2_S and F0T), K (e3v nearest neighbours; off the main line). A link is labelled by the rule that made it; a walk may budget or prune any of them, and no stored set is ever shortened.
- **Encoders** are `e3v` (defined) and `frozen` (comparison). Any read of the older encoder's cached lists or `"frozen"` source is listed in `configs/legacy_encoder_modules.json` and its numbers are called older-encoder reads.

## 3. Disk, read on 28 Nov

The laptop's `outputs/` holds 102.5 GB in 395 folders (the 45 largest are 97.5 GB). The host has 60.3 GB free of 1,021 GB (below the 90 GB clean-up point). The table is a reading of references, not a delete list: "feed refs" counts the pending or past feed rows that name the folder, "code files" the modules under `outputs/mp_unified`, `outputs/host_ops`, `scripts`, `src`, `tools` and `tests` that name it, and "pages" the `docs/*.md` pages that name it. A folder is only a candidate when no feed row names it and it is not on a hold list; each candidate still needs its own page read and a host copy check before arrays are removed (records, fits and pages always stay).

| GB | folder | tracked files | code files | feed refs | pages | reading |
|---:|---|---:|---:|---:|---:|---|
| 13.99 | `outputs/universal_v2` | 0 | 20 | 25277 | 5 | in use: feed rows name it |
| 9.10 | `outputs/u2i` | 18 | 5 | 27 | 3 | in use: feed rows name it |
| 8.86 | `outputs/m3b` | 0 | 45 | 5623 | 22 | in use: feed rows name it |
| 6.89 | `outputs/u1b` | 8 | 1 | 0 | 1 | candidate: no feed row names it; check its page and a host copy first (arrays only; records stay) |
| 6.55 | `outputs/wqk` | 17 | 21 | 0 | 12 | closed stage with code and several pages naming it: keep until its successor is declared |
| 5.01 | `outputs/stark` | 6 | 3 | 8 | 4 | in use: feed rows name it |
| 4.43 | `outputs/mp_approx_l0` | 0 | 6 | 0 | 1 | candidate: no feed row names it; check its page and a host copy first (arrays only; records stay) |
| 3.65 | `outputs/step1` | 3 | 125 | 3157 | 40 | never freed (J5S and the carves read it) |
| 2.97 | `outputs/mp_unified` | 575 | 751 | 436279 | 273 | in use: feed rows name it |
| 2.67 | `outputs/mp_approx_2wiki_anchor` | 36 | 28 | 0 | 0 | candidate: no feed row names it; check its page and a host copy first (arrays only; records stay) |
| 2.49 | `outputs/deck` | 0 | 0 | 0 | 0 | the capstone deck: never deleted |
| 2.36 | `outputs/spl1` | 20 | 7 | 1 | 4 | in use: feed rows name it |
| 2.25 | `outputs/screen` | 1047 | 191 | 3153 | 130 | in use: feed rows name it |
| 2.07 | `outputs/step4e` | 3 | 7 | 1148 | 3 | in use: feed rows name it |
| 1.98 | `outputs/u2b` | 104 | 9 | 182 | 6 | in use: feed rows name it |
| 1.76 | `outputs/u2m` | 46 | 17 | 396 | 3 | in use: feed rows name it |
| 1.75 | `outputs/enc` | 204 | 33 | 941 | 12 | in use: feed rows name it |
| 1.73 | `outputs/mp_approx_l3` | 1 | 2 | 0 | 1 | candidate: no feed row names it; check its page and a host copy first (arrays only; records stay) |
| 1.67 | `outputs/u2q` | 8 | 2 | 0 | 1 | candidate: no feed row names it; check its page and a host copy first (arrays only; records stay) |
| 1.58 | `outputs/cpu_gpu_equivalence` | 0 | 4 | 0 | 1 | candidate: no feed row names it; check its page and a host copy first (arrays only; records stay) |
| 1.47 | `outputs/u2l` | 37 | 2 | 0 | 2 | candidate: no feed row names it; check its page and a host copy first (arrays only; records stay) |
| 1.14 | `outputs/mp_approx_l1` | 0 | 2 | 0 | 1 | candidate: no feed row names it; check its page and a host copy first (arrays only; records stay) |
| 1.06 | `outputs/gpu_task_qualification` | 0 | 1 | 0 | 1 | candidate: no feed row names it; check its page and a host copy first (arrays only; records stay) |
| 1.02 | `outputs/u2r` | 14 | 4 | 4157 | 4 | in use: feed rows name it |
| 0.91 | `outputs/d83` | 3 | 4 | 0 | 1 | candidate: no feed row names it; check its page and a host copy first (arrays only; records stay) |
| 0.78 | `outputs/zrc` | 2 | 12 | 43 | 9 | never freed (J5S hold) |
| 0.78 | `outputs/rmatch` | 0 | 6 | 0 | 2 | never freed (J5S hold; frozen code) |
| 0.73 | `outputs/u2n` | 42 | 6 | 0 | 1 | candidate: no feed row names it; check its page and a host copy first (arrays only; records stay) |
| 0.64 | `outputs/u2o` | 8 | 1 | 0 | 1 | candidate: no feed row names it; check its page and a host copy first (arrays only; records stay) |
| 0.62 | `outputs/mp_approx_hotpot_anchor` | 4 | 9 | 0 | 0 | candidate: no feed row names it; check its page and a host copy first (arrays only; records stay) |
| 0.50 | `outputs/d91` | 3 | 3 | 0 | 1 | D91's folders: deletable only on the user's OK; the host's d91world holds junctions |
| 0.48 | `outputs/b1p` | 3 | 7 | 204 | 8 | in use: feed rows name it |
| 0.44 | `outputs/mp_approx_kb_anchor` | 5 | 27 | 0 | 0 | candidate: no feed row names it; check its page and a host copy first (arrays only; records stay) |
| 0.42 | `outputs/mp_approx_l4` | 0 | 2 | 0 | 1 | candidate: no feed row names it; check its page and a host copy first (arrays only; records stay) |
| 0.39 | `outputs/bench` | 16 | 11 | 301 | 17 | in use: feed rows name it |
| 0.36 | `outputs/main_table` | 0 | 1 | 0 | 1 | live stage: code, records or rows still read it |
| 0.33 | `outputs/host_ops` | 66 | 209 | 39249 | 62 | in use: feed rows name it |
| 0.33 | `outputs/sem2` | 162 | 19 | 62 | 1 | in use: feed rows name it |
| 0.22 | `outputs/step4d` | 1 | 6 | 425 | 2 | in use: feed rows name it |
| 0.22 | `outputs/step4c` | 3 | 4 | 214 | 1 | in use: feed rows name it |
| 0.21 | `outputs/b1l` | 6 | 5 | 672 | 2 | in use: feed rows name it |
| 0.19 | `outputs/deploy_ck_full_2wiki` | 0 | 1 | 0 | 0 | candidate: no feed row names it; check its page and a host copy first (arrays only; records stay) |
| 0.19 | `outputs/mp_approx_musique_anchor` | 1 | 5 | 0 | 0 | candidate: no feed row names it; check its page and a host copy first (arrays only; records stay) |
| 0.17 | `outputs/mp_approx_l6` | 0 | 1 | 0 | 1 | candidate: no feed row names it; check its page and a host copy first (arrays only; records stay) |
| 0.17 | `outputs/stk3` | 52 | 1 | 36 | 2 | in use: feed rows name it |

## 4. What was deleted on 28 Nov, and what was not

- **Deleted (host):** `outputs/spl1/models/c7`, 41 files, 0.553 GB. The model's configuration needs remote code from another repository, which the study never runs, so its weights were never loaded and SPL1 excluded it. Record: `outputs/host_cleanup/deleted_20261128_spl1_c7.json`.
- **Deleted (host), after a second look:** the eight `.npy` window arrays of `outputs/enc/e3b/win` (1.351 GB). They were first kept because no successor had been checked. The check: only `zenc3b.py` (E3B's own frozen code) names them and rebuilds them with `winenc`; `zenc3v.py` reads E3B's filed grid json and reads, `zenc6.py` reads `outputs/enc/e2s/win`; D101 adopted e3v without windows; no feed row names them. Manifests, reads and every record stay. Record: `outputs/host_cleanup/deleted_20261128_enc_e3b_win.json`.
- **Not deleted, corrected:** `outputs/enc/granitesm` (2.63 GB) and `outputs/enc/gtemodern` (5.19 GB). An earlier plan treated them as superseded encoder screens. They are the two parts the defined encoder is fused from (`zenc3v.py`: `[sqrt(0.5) v_gtemodern ; sqrt(0.5) v_granitesm]`, 1,152 wide), so they are e3v's sources, and they hold the STaRK documents' parts too. They stay.
- **Not deleted:** `outputs/leg1/vec/metaqa_all407k_partial` and `outputs/stark/amazon/splade_state.l0_cancelled.npz` (0.08 GB together; LEG1 documents the first as renamed so that an encode check ignores it), SPLADE's and K's own outputs (the user's 27 Nov ruling is to drop them from the main line first and check their return later), `outputs/spv1/` shards, the fp16 vector shards (deleted only after the int8 world exists and on the user's OK), the D91 world's junction tree, and every folder marked above as in use or on a hold list.
- **Why so little:** the one clearly superseded, rebuildable item found was the unusable model. The 28 Nov instruction covers material that is clearly superseded and rebuildable; a closed stage's arrays are neither until its successor is declared and a copy is confirmed, and the table above says which folders are candidates for that check.
