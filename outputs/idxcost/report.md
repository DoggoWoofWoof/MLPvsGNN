# The cost of the LLM-free link build: measured index-build records

Read on 2026-11-05 04:28 +0530 at HEAD ec68d2f 2026-11-05 04:08:53 +0530 by `outputs/idxcost/compile_costs.py` (`PYTHONIOENCODING=utf-8 python -I outputs/idxcost/compile_costs.py`). It reads every number below from its record at run time and lists the files it read (`report.json`, `files_read`); nothing was measured for this file. One row is one component of one corpus. There is no total across components, no per-passage rate and no extrapolation. The only arithmetic is a sum, a minimum or a maximum over the part records of one component of one corpus, marked (summed), (min) or (max); where a record holds its own number for the same thing, the script checks the two (section 8).

The components are separate. Section 2, **W2_S rule links (no spaCy)**, builds the `structural` family. Section 3, **ner family (spaCy): A6's re-run of its extraction (the original build was not timed)**, is the `ner` family, a different family of the package (`docs/U2C_POOLS_ON_CHOSEN.md` lines 21 to 23).

**Kinds.** *measured*: a clock or memory reading the run wrote. *counted*: a tally in a record. *declared*: a setting or a description written in a record, a script or a doc (a request, not a measurement). No estimated number is used. **Provenance.** `tracked, <commit>` is `git log -n1 --format='%h %ci'` of the file; `untracked` files are not in git (`outputs/` is ignored, so these records exist on the compiling machine only; this file carries their values and `report.json` their sha256). Peak figures are the records' `peak_gb`: the process's peak resident set in binary GB (kB divided by 1,048,576).

## 1. Passages per corpus

| corpus | passages | kind | record and key | provenance |
|---|---|---|---|---|
| 2wiki | 5,989,847 | counted | `outputs/u2b/2wiki/new/capstat.json` `n_nodes` | tracked, `3ca24f8 2026-10-22 03:07:39 +0530` |
| hotpotqa | 5,233,329 | counted | `outputs/u2b/hotpotqa/new/capstat.json` `n_nodes` | tracked, `3ca24f8 2026-10-22 03:07:39 +0530` |
| musique | 117,534 | counted | `outputs/u2b/musique/new/capstat.json` `n_nodes` | tracked, `3ca24f8 2026-10-22 03:07:39 +0530` |
| squad | 20,233 | counted | `outputs/u2b/squad/new/capstat.json` `n_nodes` | tracked, `3ca24f8 2026-10-22 03:07:39 +0530` |

`n_nodes` is the node count, one node per passage. The sum of `passages` over the parse parts of section 6 (untracked) equals it for each corpus, and the link rule doc's own counts for hotpotqa and 2wiki equal it (section 8).

## 2. W2_S rule links (no spaCy)

W2_S is the chosen link rule of U2b (`docs/U2B_LINK_RULE.md` line 187): variant W2, the plainest variant (the look boundary alone, lines 193 to 194), in its S form, which adds the sibling links between descriptive nodes that share a title (line 78). It reads title surfaces, a case gate, the corpus's capitalization statistic, an ambiguity guard and a cosine context choice over the frozen dense vectors (lines 43 to 44). No source file the Modal image adds mentions spaCy, and the image installs numpy==2.3.2, pyyaml==6.0.2 only (section 8). It ran in four stages (stat, merge, link, build), one container per shard, on Modal CPU containers (line 125). The result records hold no CPU count or memory limit; the CPU requests and memory below are the values declared in the Modal script's decorators, read from git (table at the end of this section). Line numbers in this section are those of `docs/U2B_LINK_RULE.md` unless another file is named.

**2a. The rule's own build.** A figure for W2 alone exists for 2wiki only: variant W2's part, the part that builds W2_S. It builds four graphs together (W2_M, W2_D, W2_S and W2_DS) in one container, so W2_S has no seconds figure of its own. hotpotqa, musique and squad have no per-variant part (their build records have no `parts` key and there is no `part_new_*.json`; before f8e1097 one container built every variant's graphs, line 151); their figures are in 2b, labelled all variants.

| component | corpus | seconds | peak GB | graphs in the container | CPU request (declared) | memory as written (declared) | record and key | provenance |
|---|---|---|---|---|---|---|---|---|
| W2_S rule links (no spaCy) | 2wiki | 480.6 | 14.29 | W2_M, W2_D, W2_S, W2_DS | 2.0 | (32768, 131072) | `outputs/u2b/2wiki/graphs/part_new_W2.json` `secs` and `peak_gb` | tracked, `c772da3 2026-10-22 12:57:09 +0530` |

*measured*: seconds and peak. *declared*: the CPU request and memory, read from the `build_part` decorator (line 420) of `outputs/mp_unified/zu2b_modal.py` at f8e1097, the commit the doc names for this build. The memory argument is (request, limit) in MiB (the installed modal 1.3.3's own comment; the version that ran is not recorded). The result records carry the rule code's hash (`zu2b_sha256`, that of `zu2b.py`), not the Modal script's, so the script version that ran is stated only for the 2wiki build.

**2b. Steps whose records cover all variants.** All variants: W2, W5e, W6e, W3e, W7e, W7, W3eg, W7eg (32 graphs: each variant in the forms D, DS, M, S). These are not the rule's own build.

| component | corpus | step | scope | seconds | peak GB | CPU request (declared) | record and key | provenance |
|---|---|---|---|---|---|---|---|---|
| W2_S rule links (no spaCy) | 2wiki | stat, one pass shared by all variants (W2, W5e, W6e, W3e, W7e, W7, W3eg, W7eg) | slowest of 32 shard containers | 527.1 | not recorded (the record holds 0) | 1.0 | `outputs/u2b/2wiki/new/capstat.json` `stat_secs_max` | tracked, `3ca24f8 2026-10-22 03:07:39 +0530` |
| W2_S rule links (no spaCy) | 2wiki | link, one pass over all variants (W2, W5e, W6e, W3e, W7e, W7, W3eg, W7eg) | slowest of 128 shard containers | 979.1 | 27.20 | 1.0 | `outputs/u2b/2wiki/graphs/build_new.json` `link_secs_max` | tracked, `c772da3 2026-10-22 12:57:09 +0530` |
| W2_S rule links (no spaCy) | 2wiki | build, all variants (W2, W5e, W6e, W3e, W7e, W7, W3eg, W7eg) | 8 build_part containers, one per variant, started by one build container (starmap); the seconds are that build container's | 2,286.9 | 19.06 | parts 2.0, build container 1.0 | `outputs/u2b/2wiki/graphs/build_new.json` `secs` | tracked, `c772da3 2026-10-22 12:57:09 +0530` |
| W2_S rule links (no spaCy) | hotpotqa | stat, one pass shared by all variants (W2, W5e, W6e, W3e, W7e, W7, W3eg, W7eg) | slowest of 32 shard containers | 344.5 | not recorded (the record holds 0) | 1.0 | `outputs/u2b/hotpotqa/new/capstat.json` `stat_secs_max` | tracked, `3ca24f8 2026-10-22 03:07:39 +0530` |
| W2_S rule links (no spaCy) | hotpotqa | link, one pass over all variants (W2, W5e, W6e, W3e, W7e, W7, W3eg, W7eg) | slowest of 128 shard containers | 701.4 | 24.27 | 1.0 | `outputs/u2b/hotpotqa/graphs/build_new.json` `link_secs_max` | tracked, `c772da3 2026-10-22 12:57:09 +0530` |
| W2_S rule links (no spaCy) | hotpotqa | build, all variants (W2, W5e, W6e, W3e, W7e, W7, W3eg, W7eg) | one container built every graph, in turn (no 'parts' key; the script before f8e1097 builds every graph in one container, `docs/U2B_LINK_RULE.md` line 151) | 3,664.2 | 9.75 | 2.0 | `outputs/u2b/hotpotqa/graphs/build_new.json` `secs` | tracked, `c772da3 2026-10-22 12:57:09 +0530` |
| W2_S rule links (no spaCy) | musique | stat, one pass shared by all variants (W2, W5e, W6e, W3e, W7e, W7, W3eg, W7eg) | the one shard container | 37.1 | not recorded (the record holds 0) | 1.0 | `outputs/u2b/musique/new/capstat.json` `stat_secs_max` | tracked, `3ca24f8 2026-10-22 03:07:39 +0530` |
| W2_S rule links (no spaCy) | musique | link, one pass over all variants (W2, W5e, W6e, W3e, W7e, W7, W3eg, W7eg) | the one shard container | 912.8 | not recorded (the record holds 0) | 1.0 | `outputs/u2b/musique/graphs/build_new.json` `link_secs_max` | tracked, `3ca24f8 2026-10-22 03:07:39 +0530` |
| W2_S rule links (no spaCy) | musique | build, all variants (W2, W5e, W6e, W3e, W7e, W7, W3eg, W7eg) | one container built every graph, in turn (no 'parts' key; the script before f8e1097 builds every graph in one container, `docs/U2B_LINK_RULE.md` line 151) | 128.3 | not recorded (the record holds null) | 2.0 | `outputs/u2b/musique/graphs/build_new.json` `secs` | tracked, `3ca24f8 2026-10-22 03:07:39 +0530` |
| W2_S rule links (no spaCy) | squad | stat, one pass shared by all variants (W2, W5e, W6e, W3e, W7e, W7, W3eg, W7eg) | the one shard container | 3.4 | not recorded (the record holds 0) | 1.0 | `outputs/u2b/squad/new/capstat.json` `stat_secs_max` | tracked, `3ca24f8 2026-10-22 03:07:39 +0530` |
| W2_S rule links (no spaCy) | squad | link, one pass over all variants (W2, W5e, W6e, W3e, W7e, W7, W3eg, W7eg) | the one shard container | 20.9 | not recorded (the record holds 0) | 1.0 | `outputs/u2b/squad/graphs/build_new.json` `link_secs_max` | tracked, `3ca24f8 2026-10-22 03:07:39 +0530` |
| W2_S rule links (no spaCy) | squad | build, all variants (W2, W5e, W6e, W3e, W7e, W7, W3eg, W7eg) | one container built every graph, in turn (no 'parts' key; the script before f8e1097 builds every graph in one container, `docs/U2B_LINK_RULE.md` line 151) | 53.4 | not recorded (the record holds null) | 2.0 | `outputs/u2b/squad/graphs/build_new.json` `secs` | tracked, `3ca24f8 2026-10-22 03:07:39 +0530` |

*measured*: seconds and peak GB (for stat and link, the largest over the shard containers); *declared*: the CPU request, one number where every committed version that could have run the step declares the same (stat and link: all 4 versions; the single-container build: the 3 versions before f8e1097). The memory arguments changed between versions and the stat, link and build records do not name the version that ran, so no memory is stated for these rows; the 2wiki build's (section 2a) is stated because the doc names its version.

- **What the seconds are.** Wall seconds in the container, not CPU seconds: from the function's `t0 = time.time()` (after the volume reload; for link, also after the selftest) to the write of its record (for `build_part`, after the container is up and the code is imported, and including the read of the link shards; for `build`, including the start of every part and the hash check of every graph file; checked, section 8). They are those of the attempt that wrote the record: stage inputs are retried (up to 3 times in the script's `STAGE`) and a retry returns an existing record without redoing the work (`docs/U2B_LINK_RULE.md` line 146); 2wiki's earlier build was preempted four times (line 152). The time lost in failed attempts is not in these records.
- **What the peak is.** The container process's peak resident set, as the record writes it (`_peak_gb`: `VmHWM`, with a `getrusage` fallback from 3ca24f8). 'Not recorded' means the record holds 0 or null; the stat records hold 0 for every corpus.

**Which link pass.**

| corpus | stat shards | link shards | link pass | basis |
|---|---|---|---|---|
| 2wiki | 32 | 128 | zu2b_fast.link_pass (inferred) | 128 link shards against 32 stat shards; the script has a link shard count of its own only from 3ca24f8, which also first calls zu2b_fast.link_pass in its link stage |
| hotpotqa | 32 | 128 | zu2b_fast.link_pass (inferred) | 128 link shards against 32 stat shards; the script has a link shard count of its own only from 3ca24f8, which also first calls zu2b_fast.link_pass in its link stage |
| musique | 1 | 1 | not stated in the record | 1 link shard, as for stat; the build record names no link pass; its link and build peaks are empty (0 or null) |
| squad | 1 | 1 | not stated in the record | 1 link shard, as for stat; the build record names no link pass; its link and build peaks are empty (0 or null) |

The link stage runs `zu2b_fast.link_pass`, which gives exactly the edges and statistics of the earlier pass, faster (`docs/U2B_LINK_RULE.md` lines 139 to 140), so a link figure from the earlier pass and one from the fast pass are not the same cost. The script calls it, and has a link shard count of its own, from 3ca24f8 on (that commit's subject: 'U2b follow-up (22 Oct, outputs unchanged): the link stage runs zu2b_fast, preemption-safe stages, 128 link shards'). The records of musique and squad do not settle which pass ran: the build record names none; one link shard equals the stat shard count, which the script before 3ca24f8 always gave (one link container per stat shard) and the script from 3ca24f8 on gives when its link shard count is left unset (`kl = k_link or k` in its `corpus` function); and their empty link and build peaks agree with the script before 3ca24f8, whose peak reader has no `getrusage` fallback, but do not prove it (section 8).

| declared in the Modal script | 6043564 (2026-10-22 01:49) | 3ca24f8 (2026-10-22 03:07) | 9f34d3f (2026-10-22 05:04) | f8e1097 (2026-10-22 05:20) |
|---|---|---|---|---|
| `stat` | cpu=1.0, memory=32768 | cpu=1.0, memory=(12288, 65536) | cpu=1.0, memory=(12288, 65536) | cpu=1.0, memory=(12288, 65536) |
| `merge` | cpu=1.0, memory=8192 | cpu=1.0, memory=8192 | cpu=1.0, memory=8192 | cpu=1.0, memory=8192 |
| `link` | cpu=1.0, memory=49152 | cpu=1.0, memory=(24576, 131072) | cpu=1.0, memory=(24576, 131072) | cpu=1.0, memory=(24576, 131072) |
| `build` | cpu=2.0, memory=98304 | cpu=2.0, memory=(32768, 131072) | cpu=2.0, memory=(32768, 131072), nonpreemptible | cpu=1.0, memory=(8192, 65536), nonpreemptible |
| `build_part` | - | - | - | cpu=2.0, memory=(32768, 131072), nonpreemptible |

Declared in `outputs/mp_unified/zu2b_modal.py` (tracked, `f8e1097 2026-10-22 05:20:59 +0530`). `memory` is one number or a (request, limit) pair, as written in the decorator. No function of any committed version passes a GPU argument; one input per container; Debian slim, Python 3.13, packages numpy==2.3.2, pyyaml==6.0.2; the image sets OPENBLAS_NUM_THREADS=8, OMP_NUM_THREADS=8, MKL_NUM_THREADS=8. The CPU model of the workers is not recorded.

## 3. ner family (spaCy): A6's re-run of its extraction (the original build was not timed)

Why this spaCy row is in the final system. The pools keep the package's spaCy-built ner family. U2l graded a spaCy-free entity family, E1 (IS alone; on a passage graph E1 together with R, the sibling links (`docs/U2L_ENTITY_MATCH.md` line 85), is W2_S array for array, line 192), and filed CHOSEN E1 (line 178); U2r graded name rules beside it and kept E1 (`docs/U2R_NAME_RULE.md` line 134, filed in 14a48ee). The pool test with E1 in the ner slot was NO_GAIN (`docs/U2M_POOLS_THREE_FAMILIES.md` section 20, line 804, filed in a2a2627 on 2026-10-27, code at fcd0f08; record `outputs/u2m/e1/verdict_e1.json`, `verdict`): on musique the I (the base and seeds plus the expansion over the three families, line 63) holds every gold (ALL, the share of questions with every gold in the pool, `docs/P1E_ANCHOR_KIND.md` line 135) for 0.7751 of the questions with the filed families (2,067.2 rows) and for 0.6838 with E1's (1,199.7 rows) (doc line 821; record `the_I.musique`). The doc reads this as musique's I reading the ner family, whose links carry bridge entities that W2_S does not reach (lines 836 to 837), and it states that the pools keep the package's NER (lines 806 to 807). U2r's model-free name rule (N1) was not adopted either: it lost 2wiki's pools by 0.0226 (`docs/NMD_NAME_LINK_DIAG.md` line 43). So the models of record read the package's spaCy-built ner family: in their pools (the models' compiled edges read all three families on every dataset, `docs/U2M_POOLS_THREE_FAMILIES.md` line 22) and in the B1 carves (carve b1 holds the NER and KNN graphs, `docs/B1H_MODELS_OF_RECORD_ON_HYPERLINKS.md` line 29; carve b1t is the NER and KNN families alone, `docs/B1L_MODELS_OF_RECORD_ON_THE_LLM_GRAPH.md` line 19; the carve's ner graph is rebuilt with spaCy, `docs/B1_HIPPORAG2_ALIGNMENT.md` lines 68 to 69). W2_S rule links (no spaCy), section 2, uses no spaCy. Superseded for the pools: `docs/U2R_NAME_RULE.md` lines 17 and 135 describe the pools' entity family as the title links alone; they date from 2026-10-26, and section 20 of the pool doc (2026-10-27) is the later record.

What the row is: spaCy NER over all passages, one CPU reserved per part, summed over parts. Hardware: the host (section 5), one CPU reserved per part (`docs/A6_LINK_WORDS_INPUT.md` line 139: '114 NER parts, 1 CPU and 1.5 GB each'; also the queue items' own field, *declared*). Wall seconds of one host job (the job's end minus its start, as the host queue recorded it), one CPU reserved per job; summed over parts they are reserved CPU-seconds (one CPU times wall seconds), not a CPU-time reading. The job runs under a retry wrapper that reruns only after a Windows file-lock error (up to 3 runs, 60 s apart), so a rerun, if one happened, is inside the job's seconds; the queue state does not say whether any did. The job also includes its own start-up (unpacking, imports, model load). Peak memory is the host job record's peak memory (the larger of its peak and live readings). Settings: spaCy 3.8.15, en_core_web_sm 3.8.0, parser, lemmatizer disabled, 9 labels, at least 3 characters, key `e.text.lower().strip()`, document frequency 2 to 25, batch size 256. These are the ner family's settings: the doc says so (`docs/A6_LINK_WORDS_INPUT.md` line 41: 'as the graph's NER family was built'), the script checks them against `ner_edges` in the package's build script (`scripts/b1_build.py` line 199, whose rule the doc describes at `docs/B1_HIPPORAG2_ALIGNMENT.md` line 68), and the doc's laptop check rebuilt every one of squad's 156,130 NER edges from A6's part, none extra (`docs/A6_LINK_WORDS_INPUT.md` lines 134 to 135). A plan figure for all 114 NER parts over six datasets is in `docs/A6_LINK_WORDS_INPUT.md` lines 140 to 141, declared before the runs; it is an estimate and is not used here.

| component | corpus | parts | CPU-seconds (job wall seconds, 1 CPU reserved per job) | per part | peak GB | merge job (one per corpus) | record and key | provenance |
|---|---|---|---|---|---|---|---|---|
| ner family (spaCy): A6's re-run of its extraction (the original build was not timed) | 2wiki | 60 | 93,280.4 (summed) | 1,224.5 (min), 1,808.5 (max) | 1.19 (max) | 277.9 s, 9.49 GB | `outputs/host_ops/hostfeed_state.json` `sent['a6-ents-2wiki-<k>of60'] -> job id; terminal[job id][3] (ran_s)` | untracked |
| ner family (spaCy): A6's re-run of its extraction (the original build was not timed) | hotpotqa | 40 | 51,449.3 (summed) | 1,159.5 (min), 1,447.3 (max) | 0.86 (max) | 233.9 s, 5.37 GB | `outputs/host_ops/hostfeed_state.json` `sent['a6-ents-hotpotqa-<k>of40'] -> job id; terminal[job id][3] (ran_s)` | untracked |
| ner family (spaCy): A6's re-run of its extraction (the original build was not timed) | musique | 2 | 1,721.7 (summed) | 856.9 (min), 864.9 (max) | 0.77 (max) | 13.6 s, 0.36 GB | `outputs/host_ops/hostfeed_state.json` `sent['a6-ents-musique-<k>of2'] -> job id; terminal[job id][3] (ran_s)` | untracked |
| ner family (spaCy): A6's re-run of its extraction (the original build was not timed) | squad | 1 | 448.5 (summed) | 448.5 (min), 448.5 (max) | 0.87 (max) | 6.0 s, 0.01 GB | `outputs/host_ops/hostfeed_state.json` `sent['a6-ents-squad-<k>of1'] -> job id; terminal[job id][3] (ran_s)` | untracked |

All *measured*, except the part counts (*counted*). The same squad extraction also ran as one part on the laptop: 314.7 s (`outputs/a6/ents/parts/squad/part_0of1.json` `seconds`, untracked, run 2026-10-17T09:41:49Z); the host job above is a separate run. Each part has one job, all done with return code 0, and none is in the host queue's retried, dropped or shed lists (section 8), so the queue state shows no failed attempt of these jobs. Semantics read at `outputs/host_ops/hostfeed2.py` (HEAD version) lines 509 and 510; items format at line 389; `outputs/host_ops/retry_cmd.py` lines 34 and 9.

## 4. The laptop name index (P4a, one thread)

Hardware: the laptop (section 5), pinned to a logical processor, one thread. The index time of the pass: the stores and both unions, the name index and the name-only mask, the question texts, the walk buffers and FastPool's stamps, built once per corpus and recorded, not part of the timed per-question passes (`docs/P4A_POOL_COLD.md` lines 52 to 55; the laptop, its pin and one thread at lines 60 and 61). The record's own total is setup_total_s; name_index_s and name_only_s are two of its itemised parts (the record also itemises pools_file_s, lists_and_pools_s and buffers_s); they are not re-summed here.

| corpus | setup_total_s | name_only_s (the name-only mask) | name_index_s (the name index) | run | record and key | provenance |
|---|---|---|---|---|---|---|
| 2wiki | 111.65 | 42.36 | 64.53 | 1 thread, logical processor 2 | `outputs/p4a/2wiki.json` `index.setup_total_s` | tracked, `056918b 2026-10-14 11:56:16 +0530` |
| hotpotqa | 131.56 | 47.84 | 78.50 | 1 thread, logical processor 2 | `outputs/p4a/hotpotqa.json` `index.setup_total_s` | tracked, `056918b 2026-10-14 11:56:16 +0530` |
| musique | 8.28 | 1.53 | 4.04 | 1 thread, logical processor 2 | `outputs/p4a/musique.json` `index.setup_total_s` | tracked, `056918b 2026-10-14 11:56:16 +0530` |
| squad | 4.53 | 0.33 | 0.12 | 1 thread, logical processor 2 | `outputs/p4a/squad.json` `index.setup_total_s` | tracked, `056918b 2026-10-14 11:56:16 +0530` |

*measured* (seconds, rounded to two decimals here; `report.json` holds the record's values); the thread and pin settings are *declared*.

## 5. Hardware lines

- Host, CPU: Intel i9-14900, 32 threads, AVX2 (`docs/GPU_HOST_HANDOVER.md` line 24 (row 'CPU'); tracked, `016b850 2026-10-03 23:14:34 +0530`; *declared*)
- Host, memory: 127.7 GB (`docs/GPU_HOST_HANDOVER.md` line 25 (row 'memory'); tracked, `016b850 2026-10-03 23:14:34 +0530`; *declared*)
- Host, GPU: NVIDIA RTX 4500 Ada Generation, 24 GB, driver 596.71 (`docs/GPU_HOST_HANDOVER.md` line 26 (row 'GPU'); tracked, `016b850 2026-10-03 23:14:34 +0530`; *declared*)
- Host, scheduler capacity: 32 CPUs minus 2 reserved, 127.7 GB minus 8 reserved, GPU 0 (`docs/GPU_HOST_HANDOVER.md` line 31 (row 'scheduler capacity'); tracked, `016b850 2026-10-03 23:14:34 +0530`; *declared*)
- Host, machine: a shared lab workstation; other accounts use it (`docs/GPU_HOST_HANDOVER.md` line 22 (row 'machine', the text after the host's name); tracked, `016b850 2026-10-03 23:14:34 +0530`; *declared*)
- Host, used for: A6's NER part jobs (one CPU reserved per job, no GPU named in their queue items) and, beside, P2c part 2's parse parts.
- Laptop: Intel Family 6 Model 186, 12 logical CPUs, AVX2, 15.7 GB of memory (`docs/GPU_HOST_HANDOVER.md` line 35; tracked, `016b850 2026-10-03 23:14:34 +0530`; *declared*). Used for: P4a's index setup (pinned, one thread).
- Modal CPU containers: one container per shard (`docs/U2B_LINK_RULE.md` line 125); the CPU request and memory are the declared values of section 2, and no result record states them; wall-clock seconds in the container (time.time differences), not CPU seconds; the CPU model of the workers is not recorded.

## 6. Beside, not in the final system: P2c part 2's parse

Not in the final system: `docs/P2C2_FOURTH_FAMILY.md` line 91 says 'NO_GAIN: the fourth family is not adopted, and no full run follows'. The fourth family's parse is part 1's version 1, run on the host over every passage (`docs/P2C2_FOURTH_FAMILY.md` line 19), with the full spaCy pipeline including its dependency parser (`docs/P2C_LLM_FREE_TRIPLES.md` line 63). Hardware: the host (the part files' `where` says `host`). *measured*, *summed* over the part files; the peak is the maximum over parts.

| corpus | parts | seconds | per part | peak GB | record and key | provenance |
|---|---|---|---|---|---|---|
| 2wiki | 60 | 115,407.0 (summed) | 1,190.7 (min), 2,414.6 (max) | 1.39 (max) | `outputs/p2c/parse_2wiki_<k>of60.json` `seconds` | untracked |
| hotpotqa | 40 | 60,386.3 (summed) | 1,293.3 (min), 1,681.8 (max) | 0.98 (max) | `outputs/p2c/parse_hotpotqa_<k>of40.json` `seconds` | untracked |
| musique | 2 | 2,005.7 (summed) | 980.8 (min), 1,024.9 (max) | 0.92 (max) | `outputs/p2c/parse_musique_<k>of2.json` `seconds` | untracked |
| squad | 1 | 592.3 (summed) | 592.3 (min), 592.3 (max) | 0.97 (max) | `outputs/p2c/parse_squad_<k>of1.json` `seconds` | untracked |

## 7. The gaps

- **embeddings (the frozen dense encoder, gte-Qwen2-1.5B), every corpus**: no wall time of any encode run is in the records compiled here. The encoder is named at `docs/B1_HIPPORAG2_ALIGNMENT.md` line 138. `docs/M3A_WEBQSP_ENCODE_BUDGET.md` line 48 says 'no shard status JSON in the package records a timing -- they carry rows, sha, dtype and dim only' (the doc's subject is one corpus; its encode rate is declared and marked DECLARED_NOT_MEASURED, line 53, and none of its figures is used here). The package's own logs, if they hold a timing, were not searched.
- **the original ner and knn builds, every corpus**: ner: no duration in any of the four corpora's graph manifests in the frozen sibling package. knn: a duration for musique (513.4 s, key families.knn.source.source_manifest.build_seconds); a build timestamp and no duration for 2wiki (2026-09-10T01:54:51), hotpotqa (2026-09-10T00:13:28); neither a duration nor a timestamp for squad. Read at run time from the sibling package's `data/final_canonical/<corpus>/graph/GRAPH_MANIFEST.json` (untracked there; its data folder is git-ignored), by scanning the ner and knn family entries for a seconds-like key. That manifest's knn method line says: EXHAUSTIVE_FP32_COMPUTE over RESOLVED canonical vectors (local CPU). The package's other records and logs were not searched. Section 3 re-runs the ner family's extraction; it is not the original build, and nothing re-runs the knn build.
- **the CSR family stores (`outputs/m3b/csr/<corpus>_<family>.npz`)**: no build time of its own is recorded: a missing store is built by whatever job first opens it, and no record read here separates that time from the job's. `src/mp_retrieval/m3b_pools.py` lines 155 to 163: load_or_build_store builds a missing store and saves it, and takes no timing (checked at run time). The stores are read from `outputs/m3b/csr` through it (`docs/STEP4_POOL_REACH.md` line 38); the 12 stores of the four corpora (three families each) are present in this working tree, untracked.
- **Modal container-second totals (stat, link, build) and the merge stage**: only the slowest shard's seconds per stage are kept, not the sum over shard containers; the merge stage's seconds are in no record; the time lost in failed or preempted attempts is in none either. The Modal script writes each shard's record to the Modal volume, and this repository holds the merged capstat.json and the build record only (both checked at run time).
- **W2 alone, where there is no part record**: the build seconds of W2 for hotpotqa, musique and squad (no per-variant parts; the figures are for all variants), and the stat and link seconds of W2 alone for every corpus (one pass serves all variants). Section 2; only 2wiki has a W2 part record, and that part builds four graphs together, so W2_S alone has no figure there either.
- **peak memory of some Modal steps**: not recorded (the record holds 0 or null) for 2wiki stat, hotpotqa stat, musique stat, musique link, musique build, squad stat, squad link and squad build. Marked 'not recorded' in the rows of section 2.
- **the Modal workers' CPU model**: not recorded; the result records hold neither CPU count nor memory, and the script's decorators declare the CPU request and memory only. Section 5.
- **the Modal dense stage (the frozen vectors the rule's context choice reads, placed on the volume)**: no seconds: its records (dense_verified.json and dense_reread.json, per corpus) hold dataset, manifest_sha256, n_rows, shards, status only. Keys checked at run time in all eight files.
- **A6's per-part files for 2wiki, hotpotqa and musique**: this working tree holds the A6 part file of squad only (a laptop run); for the other corpora the host queue's job seconds are used, which include each job's own start-up and any rerun by the retry wrapper. Section 3.
- **a per-corpus end-to-end index figure**: not recorded and not computed here, by design: the components have different hardware and different stages. No total appears in this record.

## 8. What the script checked

- ok: 2wiki: the parse parts are k = 0..59 of 60, each naming itself (60 files)
- ok: 2wiki: every parse part ran on the host (key where)
- ok: hotpotqa: the parse parts are k = 0..39 of 40, each naming itself (40 files)
- ok: hotpotqa: every parse part ran on the host (key where)
- ok: musique: the parse parts are k = 0..1 of 2, each naming itself (2 files)
- ok: musique: every parse part ran on the host (key where)
- ok: squad: the parse parts are k = 0..0 of 1, each naming itself (1 file)
- ok: squad: every parse part ran on the host (key where)
- ok: 2wiki: n_nodes equals the sum of passages over the parse parts (5989847 and 5989847)
- ok: hotpotqa: n_nodes equals the sum of passages over the parse parts (5233329 and 5233329)
- ok: musique: n_nodes equals the sum of passages over the parse parts (117534 and 117534)
- ok: squad: n_nodes equals the sum of passages over the parse parts (20233 and 20233)
- ok: hotpotqa and 2wiki: the counts in the link rule doc equal n_nodes (`docs/U2B_LINK_RULE.md` line 134)
- ok: 2wiki: the stat and build records name the rule code the working tree holds (sha256 of zu2b.py, u1b.py, u1d.py) (zu2b.py 4bfbf7479a72, u1b.py cb485a77ea63, u1d.py c4b62ce8b69f)
- ok: hotpotqa: the stat and build records name the rule code the working tree holds (sha256 of zu2b.py, u1b.py, u1d.py) (zu2b.py 4bfbf7479a72, u1b.py cb485a77ea63, u1d.py c4b62ce8b69f)
- ok: musique: the stat and build records name the rule code the working tree holds (sha256 of zu2b.py, u1b.py, u1d.py) (zu2b.py 4bfbf7479a72, u1b.py cb485a77ea63, u1d.py c4b62ce8b69f)
- ok: squad: the stat and build records name the rule code the working tree holds (sha256 of zu2b.py, u1b.py, u1d.py) (zu2b.py 4bfbf7479a72, u1b.py cb485a77ea63, u1d.py c4b62ce8b69f)
- ok: no source file the Modal image adds mentions spaCy (`outputs/mp_unified/u1b.py`, `outputs/mp_unified/u1d.py`, `outputs/mp_unified/zu2b.py`, `outputs/mp_unified/zu2b_fast.py`, `src/mp_retrieval/m3b_pools.py`)
- ok: the Modal image installs no spaCy (numpy==2.3.2, pyyaml==6.0.2)
- ok: the Modal image sets OPENBLAS_NUM_THREADS, OMP_NUM_THREADS and MKL_NUM_THREADS, all to one value (OPENBLAS_NUM_THREADS=8, OMP_NUM_THREADS=8, MKL_NUM_THREADS=8)
- ok: no function of any committed Modal script passes a gpu argument, and the stage and orchestrator settings name none (4 versions)
- ok: the separate link shard count and the zu2b_fast link pass enter the script in one commit, and stay (the separate count first at 3ca24f8, the zu2b_fast call first at 3ca24f8)
- ok: an unset link shard count equals the stat shard count: one link container per stat shard before the separate count, and kl = k_link or k (k_link defaults to 0) from it on (function corpus of `outputs/mp_unified/zu2b_modal.py` at 6043564 and at 3ca24f8)
- ok: in every committed version, stat, link, build and build_part start their clock after the volume reload and write their seconds as the time since that start (4 versions)
- ok: link's seconds start after zu2b_fast's selftest (`outputs/mp_unified/zu2b_modal.py` at f8e1097)
- ok: build_part's seconds run from after the container is up and the code is imported, through reading the link shards, to the end of the part (`outputs/mp_unified/zu2b_modal.py` at f8e1097, function build_part)
- ok: the build container's seconds include starting every part and checking every graph file's hash (`outputs/mp_unified/zu2b_modal.py` at f8e1097, function build)
- ok: the build record gets its 'parts' key from the commit the doc names, and not before (first at f8e1097)
- ok: the commit the doc names states the layouts in its own comment (`outputs/mp_unified/zu2b_modal.py` at f8e1097)
- ok: 2wiki: the stat and build records hold no CPU count or memory key (no key matches cpu, mem or core)
- ok: 2wiki: the stat record holds no merge seconds (capstat.json keys)
- ok: 2wiki: the link record names the shard count the doc states (128 link shard hashes; `docs/U2B_LINK_RULE.md` line 148 says 128)
- ok: 2wiki: the W2 part file equals the build record's parts.W2 (480.6 s, 14.29 GB)
- ok: 2wiki: the W2 part builds the graph W2_S (W2_M, W2_D, W2_S, W2_DS)
- ok: 2wiki: the build stage's seconds are at least its slowest part's (2286.9 and 2038.4)
- ok: 2wiki: the build record's parts are its variants (['W2', 'W5e', 'W6e', 'W3e', 'W7e', 'W7', 'W3eg', 'W7eg'])
- ok: 2wiki: the doc's build line rounds the record's secs and peak_gb (2286.9 s, 19.06 GB; `docs/U2B_LINK_RULE.md` line 229)
- ok: hotpotqa: the stat and build records hold no CPU count or memory key (no key matches cpu, mem or core)
- ok: hotpotqa: the stat record holds no merge seconds (capstat.json keys)
- ok: hotpotqa: the link record names the shard count the doc states (128 link shard hashes; `docs/U2B_LINK_RULE.md` line 148 says 128)
- ok: hotpotqa: the build record has no per-variant parts (key parts absent)
- ok: musique: the stat and build records hold no CPU count or memory key (no key matches cpu, mem or core)
- ok: musique: the stat record holds no merge seconds (capstat.json keys)
- ok: musique: the build record has no per-variant parts (key parts absent)
- ok: musique: the peaks are empty and the script before 3ca24f8 has no peak fallback (link_peak_gb_max 0, peak_gb null)
- ok: squad: the stat and build records hold no CPU count or memory key (no key matches cpu, mem or core)
- ok: squad: the stat record holds no merge seconds (capstat.json keys)
- ok: squad: the build record has no per-variant parts (key parts absent)
- ok: squad: the peaks are empty and the script before 3ca24f8 has no peak fallback (link_peak_gb_max 0, peak_gb null)
- ok: squad's laptop part covers the corpus's passages (20233 and 20233)
- ok: the doc's laptop check names the squad part file's mentions and keys, and its seconds rounded (`docs/A6_LINK_WORDS_INPUT.md` line 134: 188,101 mentions, 64,439 keys, 315 s; the part file: 188101, 64439, 314.7)
- ok: A6's settings are the ner family's: labels, document-frequency range, minimum characters, key, batch size and disabled components equal those of ner_edges in the package's build script (`scripts/b1_build.py` line 199; `docs/B1_HIPPORAG2_ALIGNMENT.md` line 68)
- ok: the retry wrapper reruns only after a Windows file-lock error (`outputs/host_ops/retry_cmd.py` lines 34, 9)
- ok: 2wiki: A6's parts are k = 0..59 of 60, as the doc's part count (60 jobs; `docs/A6_LINK_WORDS_INPUT.md` line 43)
- ok: 2wiki: every A6 part job is done with return code 0 (terminal[job id][0] and [1])
- ok: 2wiki: A6's merge job is done with return code 0 (a6-merge-2wiki)
- ok: 2wiki: each A6 job id carries its job's name, and no A6 job is in the queue's retried, dropped or shed lists (61 jobs)
- ok: 2wiki: every A6 part's queue item asks for 1 CPU and 1.5 GB, names no GPU, and runs part k/n of the corpus on the host under the retry wrapper (3 runs, 60 s apart) (`outputs/host_ops/hostfeed_items.txt` (HEAD version), 60 item lines)
- ok: hotpotqa: A6's parts are k = 0..39 of 40, as the doc's part count (40 jobs; `docs/A6_LINK_WORDS_INPUT.md` line 43)
- ok: hotpotqa: every A6 part job is done with return code 0 (terminal[job id][0] and [1])
- ok: hotpotqa: A6's merge job is done with return code 0 (a6-merge-hotpotqa)
- ok: hotpotqa: each A6 job id carries its job's name, and no A6 job is in the queue's retried, dropped or shed lists (41 jobs)
- ok: hotpotqa: every A6 part's queue item asks for 1 CPU and 1.5 GB, names no GPU, and runs part k/n of the corpus on the host under the retry wrapper (3 runs, 60 s apart) (`outputs/host_ops/hostfeed_items.txt` (HEAD version), 40 item lines)
- ok: musique: A6's parts are k = 0..1 of 2, as the doc's part count (2 jobs; `docs/A6_LINK_WORDS_INPUT.md` line 43)
- ok: musique: every A6 part job is done with return code 0 (terminal[job id][0] and [1])
- ok: musique: A6's merge job is done with return code 0 (a6-merge-musique)
- ok: musique: each A6 job id carries its job's name, and no A6 job is in the queue's retried, dropped or shed lists (3 jobs)
- ok: musique: every A6 part's queue item asks for 1 CPU and 1.5 GB, names no GPU, and runs part k/n of the corpus on the host under the retry wrapper (3 runs, 60 s apart) (`outputs/host_ops/hostfeed_items.txt` (HEAD version), 2 item lines)
- ok: squad: A6's parts are k = 0..0 of 1, as the doc's part count (1 job; `docs/A6_LINK_WORDS_INPUT.md` line 43)
- ok: squad: every A6 part job is done with return code 0 (terminal[job id][0] and [1])
- ok: squad: A6's merge job is done with return code 0 (a6-merge-squad)
- ok: squad: each A6 job id carries its job's name, and no A6 job is in the queue's retried, dropped or shed lists (2 jobs)
- ok: squad: every A6 part's queue item asks for 1 CPU and 1.5 GB, names no GPU, and runs part k/n of the corpus on the host under the retry wrapper (3 runs, 60 s apart) (`outputs/host_ops/hostfeed_items.txt` (HEAD version), 1 item line)
- ok: the E1 verdict record says NO_GAIN, and its musique I figures equal the doc's table (`outputs/u2m/e1/verdict_e1.json`: [0.7751, 2067.2, 0.6838, 1199.7]; `docs/U2M_POOLS_THREE_FAMILIES.md` line 821)
- ok: the E1 verdict record was committed with the filing (a2a2627 2026-10-27 03:27:21 +0530 and a2a2627)
- ok: the code commit the doc names changed `outputs/mp_unified/zu2m_e1.py`, and is older than the filing commit (fcd0f08 2026-10-26 12:46:09 +0530; filing a2a2627 2026-10-27 03:27:21 +0530)
- ok: U2r's filing is older than U2m's section 20 filing (14a48ee 2026-10-26 08:45:47 +0530; a2a2627 2026-10-27 03:27:21 +0530)
- ok: the B1H doc's carve b1 holds the NER and KNN graphs (`docs/B1H_MODELS_OF_RECORD_ON_HYPERLINKS.md` line 29)
- ok: the B1 doc's ner rule is the spaCy rule, two lines on (`docs/B1_HIPPORAG2_ALIGNMENT.md` lines 68, 69)
- ok: the doc's index-time paragraph is lines 52 to 55 (`docs/P4A_POOL_COLD.md`)
- ok: 2wiki: P4a ran one thread, pinned to a logical processor (`outputs/p4a/2wiki.json`)
- ok: 2wiki: the record's setup_total_s is at least each itemised part it lists (111.6504454612732 and 64.52541065216064)
- ok: hotpotqa: P4a ran one thread, pinned to a logical processor (`outputs/p4a/hotpotqa.json`)
- ok: hotpotqa: the record's setup_total_s is at least each itemised part it lists (131.56149339675903 and 78.4966173171997)
- ok: musique: P4a ran one thread, pinned to a logical processor (`outputs/p4a/musique.json`)
- ok: musique: the record's setup_total_s is at least each itemised part it lists (8.28215217590332 and 4.040663957595825)
- ok: squad: P4a ran one thread, pinned to a logical processor (`outputs/p4a/squad.json`)
- ok: squad: the record's setup_total_s is at least each itemised part it lists (4.532017946243286 and 3.606292486190796)
- ok: the host and laptop lines sit within lines 21 to 35 of the handover doc (cpu line 24, memory line 25, gpu line 26, scheduler_capacity line 31, machine line 22, laptop line 35)
- ok: load_or_build_store builds a missing store and takes no timing (`src/mp_retrieval/m3b_pools.py` lines 155 to 163)
- ok: the stores of the four corpora are present in this working tree and none is tracked (12 of 12)
- ok: no per-shard stat or link record is kept in this repository, though the script writes one per shard (`outputs/u2b/2wiki/new` holds capstat.json; `outputs/u2b/hotpotqa/new` holds capstat.json; `outputs/u2b/musique/new` holds capstat.json; `outputs/u2b/squad/new` holds capstat.json)
- ok: the Modal dense stage's records hold no seconds key, and all eight hold the same keys (dataset, manifest_sha256, n_rows, shards, status in all 8 files)
