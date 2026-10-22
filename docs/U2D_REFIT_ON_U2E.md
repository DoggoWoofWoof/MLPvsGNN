# U2d: the lean pair and the free GNN refit on U2e's pools and the LLM-free graph

Declared 22 October 2026, after U2e's verdict (RAISED, cf2bb06) and before any U2d number. Code:
`outputs/mp_unified/zu2d_pools.py`, `outputs/mp_unified/look_u2d.py`, `outputs/mp_unified/zu2d.py` and
`outputs/mp_unified/zu2d_modal.py`. Everything else is imported unchanged. Numbers here are development numbers.

The host has been offline since 20 October. On 22 October the user said: "use modal for the refit, keep going". So
U2d runs on Modal, end to end.

## 1. The question

U2e picked `kt_cos:ada_f12_k0@0.85` on U2c's chosen graph (W2_S, the LLM-free links). On every training dataset's
s1sel it holds every gold for at least 0.955 of the questions, at 1,812.6 rows (P1f's question-weighted mean). Today's
pools (P1c2's) cost 699.7 rows on the same measure.

| dataset | U2e's pick: ALL | rows |
| --- | --- | --- |
| metaqa | 0.962 | 4,384 |
| squad | 0.979 | 50 |
| musique | 0.964 | 5,437 |
| hotpotqa | 0.956 | 561 |
| 2wiki | 0.967 | 840 |
| webqsp (s1eval, beside) | 0.893 | 4,755 |

U2d asks two questions:
1. **The pools.** Do the models rank better when they are trained and read on these pools and this graph, in domain
   and zero-shot?
2. **"via".** Does it help a model to know how each row came into the pool? This is the user's question of
   22 October: "if the models know what caused this node to come into the pool it might be able to rank better".

There are three models, each unchanged:
- **zrckp**, the MLP: zrc's model plus the pointwise KT head (zp1c3p). It uses no message passing.
- **zspk**, the GNN of the lean pair: zsp's model plus the KT head (zp1c3).
- **zg6**, the free GNN: G6's own encoder, hidden-state message passing and readout (zg6, docs/G6_FREE_GNN.md).

## 2. The two builds

U2d builds two sets of pools on P1c2's eleven carves: each training dataset's fit carve (musique's s1fit) and the six
s1eval carves. Both builds and all their fits run on Modal.

**base: today's pools on today's graph, rebuilt.** All U2d fits run on one Modal GPU type, and Modal's floats need not
match the host's. A base built and fit on the same device keeps the device out of every comparison. The base also
carries U2d's seed null (section 6).

Per question:
- B is the base and seeds;
- KIND_T's order comes from P1c2's `kind_t` (P1e's orders) on today's graph;
- the expansion is the order's first |I_q| − |B| nodes outside B. This is P1c2's rule, unchanged.

Checks (any failure stops the item):
- **Per question**, on the seven carves whose P1c2 files are on the laptop (the six s1eval carves and musique's
  s1fit): B, |I_q|, the frozen pool's sha, the expansion, and the entity and route flags equal P1c2's filed arrays.
- **Per carve**, on all eleven (the merge): the counts equal P1c2's manifest, field by field: questions, changed,
  short, |I_q| − |B| (mean and maximum), the mean frozen pool, the entity and route shares, and the ALL and recall of
  I and of P1c2's pools. On the four fit carves whose P1c2 files are on the host only, this is the check.
- **The rows** (the gate, on every carve of both builds): |I_q| on today's graph is step 1's. For each step 1 cache
  part, the sha256 of its questions' int32 sizes, saved as step 1 saves them, equals the part record's n array.
- **Beside** (it decides nothing): the KT input rebuilt on the base build is compared with P1c3's filed KT records
  (the sha256 of the kt and n arrays) on all eleven carves, and each carve is reported REPRODUCED or DIFFERS.

**new: U2e's pick on W2_S.** W2_S replaces the structural links on musique, squad, hotpotqa and 2wiki, exactly as
U2c and U2e swap it (`zu2c.graph_csr` and `zu2c.csr_store`, unchanged). metaqa and webqsp keep their KB. Per
question, as in U2e's stats (section 4 of docs/U2E_POOLS_TO_THE_CEILING.md):
- I^G is the frozen pool prepared on W2_S;
- full is KIND_T's order on W2_S, outside B;
- fill is full's first |I^G_q| − |B| nodes;
- N1, R0 and I outside B come from D5b's `n1_table`, P1d's sources and I^G;
- cos_0 and kt_cos come from `zu2e.question_orders`, unchanged, with no second link;
- k = min(|kt_cos|, round(12 × |fill|)), from `zu2e.budget_k`;
- keep is P1h's cut at τ = 0.85 (`zp1h.keep_counts`);
- the expansion is kt_cos's first keep nodes.

Checks:
- B and the seeds are the same on both graphs, and B is step 4d's (U2e's checks). The graph's npz sha256 is U2b's
  build record's, and every question's orders pass `zu2e.orders_exact`.
- **A check-only run on the five training datasets' s1sel.** It writes nothing but its check records. On every
  question, |B|, |fill|, |I|, |kt_cos|, |cos_0| and every gold's place in kt_cos and in cos_0 must equal U2e's filed
  stats arrays. The check-merge then requires the pick's summary (ALL, recall, mean rows and p90 rows) to equal U2e's
  verdict.
- **webqsp s1eval** is both U2e's carve and U2d's. The same per-question check runs inside its pools stage, and its
  merge requires the same summary to equal U2e's verdict.

**The pools files** use P1g-b's format (`p1gb_pools.pool_arrays`): ids, the frozen pool's size and sha on the build's
graph, B, the expansion and its counts. Three columns are added:
- `today_inc_size`: |I_q| on today's graph, which the gate checks against step 1's rows;
- `budget`: k, the order's budget capped at the order's length, before the cut: min(|kt_cos|, round(12 × |fill|)) on
  the new build, min(|full|, |I_q| − |B|) on the base;
- `kt_len`: |kt_cos| (new) or |full| (base).

On the new build, fill_want is round(12 × |fill|), fill_k is keep and x_count is 0. On the base build, fill_want is
|I_q| − |B| and fill_k is the expansion's length. In both, "short" counts the questions whose expansion holds fewer
nodes than fill_want (fill_k < fill_want, P1g-b's count).

**Reported beside, from the pools stage:**
- each carve's ALL, gold recall and mean rows;
- the R@5 cap, the mean over questions of min(5, the golds the pool holds) / |golds|;
- the hit@1 cap, the share of questions whose pool holds a gold.

## 3. The features

Each build has its own roots, `outputs/u2d/<build>/` (pools, look, cache, chains, links, kt, gate.json). Everything is
the filed pipeline, unchanged:

- **Looks** (`look_u2d.py --build B`): look_x_six after `m3b_compile.prepare`. On the new build, W2_S goes into the
  context's structural store before prepare, as look_u1c swaps it. The frozen pool, the look's edges and everything
  built from them therefore read W2_S. Each question's pool becomes B plus the filed expansion, through
  `p1gb_pools.replace_pools`, unchanged. Each look shard files a pools record naming the pools file, its counts and
  the graph. A pool above 32,767 nodes stops the look: the look's edge arrays index a pool in int16. U2e's pick
  reaches at most 9,680 nodes on musique's s1sel and 9,688 on webqsp's s1eval.
- **Caches:** lean_cache on the build's looks, with step 1's bases (basis_2wiki, basis_hotpotqa) copied in.
- **Chains and links:** rmatch's and zrc's chains on metaqa and webqsp, and zlink's pool edges on every carve, through
  zp1c2's build on the build's roots. The links are the looks' edges, so they are W2_S's on the new build.
- **Relation tables:** today's on both builds. The KB does not change, and the passage links carry no relation, as
  today (`zu2c.csr_store`).
- **The KT input:** P1c3's eight columns (`p1c3_kt.columns`, unchanged), from the six orders of P1e's `p1e_orders` on
  the build's graph. The fill column marks the expansion, every row outside B, as on P1c2's pools.
  - The pools stage computes the orders anyway, so it writes each question's columns over its pool sorted by node.
  - The KT stage puts them in the cache's row order. It checks that each question's cache rows are its pool, each
    node once.
  - The columns are row by row, so this is the same as computing them on the cache's rows.
- **The via input** (new build only): section 4.
- **The gate** (`zu2d_pools.py gate --build B`), on every carve before any fit:
  - the pools file is the manifest's, and one rule built every carve;
  - the cache's questions are step 1's and the file's;
  - step 1's rows are the filed `today_inc_size` (by the n array's sha256 in step 1's record);
  - the cache's rows are the filed pool sizes;
  - every look shard filed a pools record for this file;
  - the KT (and via) records sit on the cache's parts and root.

## 4. The via columns

The new build's pools come from three sources: KIND_T's order, B's one-link neighbours by the question's cosine, and
the first stage's lists. Seven columns per row record which of them brought the row in, and how strongly. The pools
stage computes them when it builds the pool, before any row is scored. They use no gold, read no score, and are not
normalised over the question's rows. Rows of B take the value of "not from this source".

| column | value |
| --- | --- |
| `ktcos` | the row's 0-based place p in kt_cos, as log1p(p) / log1p(2^20), at most 1; 1 when kt_cos does not hold the row |
| `cos` | the same for cos_0 (U_0 by the question's cosine) |
| `n1` | 1 when the row is one link from B (D5b's N1 over the three link families), else 0 |
| `n1cnt` | log1p(the number of B nodes linked to the row) / log1p(64), at most 1 |
| `n1w` | log1p(30.5 × the summed first-stage weight of those B nodes) / log1p(64), at most 1. The weight is P1e's steer, 1/(60 + dense rank) + 1/(60 + SPLADE rank), so 30.5 × it is 1 for a node ranked first in both lists |
| `r0` | 1 when the first stage's lists (P1d's R0, 1,000 deep each) hold the row |
| `inI` | 1 when the frozen pool I^G holds the row outside B |

They are written as `via.npy` (rows × 7, float16) beside `kt.npy`, in the cache's row order, with their own record.

Two arms read them, each the filed head widened to take them:
- **zrckpv:** zrckp's pointwise head over 15 inputs (8 KT + 7 via): 16 × 15 + 16 + 16 + 1 = 273 weights;
- **zspkv:** zspk's head over 16 inputs (8 KT + 7 via + the row's z-scored score): 289 weights.

As in both filed heads, the first layer comes from the head's own generator (seed + 3301), and the last layer starts
at zero. At the start, each forward is zrc's or zsp's bit for bit, and the global generator is where that model
leaves it.

Flagged, as for every pool so far: the pool and these columns read the graph. They are fixed before any model runs, and
both models get the same columns, so the MLP's own scoring stays free of message passing.

## 5. The fits

All fits use the card settings: variant p, config `2e-3:1e-4:0.1:8:2`, hidden 128, the split's fit carves and basis,
and threads 2. The splits are L-musique and L-hotpotqa. Each fit is read with p@swa on the six s1eval carves of its own
build.

| build | arms | seeds | fits |
| --- | --- | --- | --- |
| base | zrckp, zspk, zg6 | 0 | 6 |
| base | zrckp, zspk | 1 and 2 (the null) | 8 |
| new | zrckp, zspk, zg6, zrckpv, zspkv | 0 | 10 |

That is 24 fits and 24 reads. They are named `scr-u2d-<build>-<arm>[-s<seed>][-hp]`; for example
`scr-u2d-new-zspkv-hp` is zspkv's L-hotpotqa fit on the new build.

**One device.** Every fit and read runs on an NVIDIA L40S on Modal, from one pinned image (torch 2.8.0 with CUDA), with
the filed deterministic flags (`lean_gpu.set_flags`, CUBLAS_WORKSPACE_CONFIG `:4096:8`). Each fit records the GPU's
name and the image. A comparison between two fits from different devices or images stops. The device is five fields:
the GPU's name, the image's tag (`u2d-gpu:py3.13:torch2.8.0+cu128:numpy2.3.2`, which the GPU image sets and zu2d.py
requires), torch, CUDA and cuDNN. Modal's own image id differs between workspaces (each builds the same image itself),
so it is recorded beside and not compared. A read too large for the
card may be re-run with `--rows-cap` (whole questions per batch; the scores do not change), recorded in its read.json
as before.

## 6. The null

U2d has its own seed null, on the base build. For zrckp and zspk, the fits with seeds 1 and 2 are each compared with
the same arm's seed-0 card fit, on each split and dataset. That gives two seed-only differences, D1 and D2, per read.
The floor is docs/SCREENS.md's: max(0.0075, 2 × sqrt((D1² + D2²) / 2)).
- zrckp's null calls the MLP arms (zrckp, zrckpv).
- zspk's null calls the GNN arms (zspk, zspkv, zg6). G6's rule also takes zspk's seeds.

Where a pair's base is not the null's seed-0 fit, the null is re-based as relz re-bases it: through the comparison of
that base with the null's seed 0. The floors do not change.

## 7. The decisions

R@5 is over the question's golds, so a gold the pool misses counts as a miss. GAIN, LOSS and WITHIN are
docs/SCREENS.md's:
- a GAIN needs delta R@5 ≥ 0.0075 and the 2,000-resample question bootstrap's 95% interval above 0;
- a LOSS is the mirror;
- the re-call turns a GAIN or a LOSS whose |delta| is below the read's floor into WITHIN.

**The rule, fixed before any number** (P1g-b's, unchanged), for a decision over its arms:
- **ADOPT_FOR_FULL_RUN:** no arm's re-call has a LOSS on any of its twelve reads, and the arms together have at least
  one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** a re-call is missing, is not U2d's, does not hold twelve reads, or compares fits from different
  devices.

**The four decisions:**

| decision | what it asks | arms (new fit against base fit) |
| --- | --- | --- |
| A | the pools, for the lean pair | zrckp new vs zrckp base; zspk new vs zspk base |
| B | via, on the new pools | zrckpv new vs zrckp new; zspkv new vs zspk new |
| C | the pools and via together | zrckpv new vs zrckp base; zspkv new vs zspk base |
| G | the pools, for zg6 | zg6 new vs zg6 base (one arm) |

**What goes to the lean pair's full run:**

| A | B | C | the full run |
| --- | --- | --- | --- |
| ADOPT | ADOPT | (beside) | U2e's pools with via (zrckpv, zspkv) |
| ADOPT | NOT | (beside) | U2e's pools (zrckp, zspk) |
| NOT | ADOPT | ADOPT | U2e's pools with via |
| NOT | ADOPT | NOT | none: the lean pair stays on P1c2's pools |
| NOT | NOT | (beside) | none: the lean pair stays on P1c2's pools |

C is always computed. It decides only when A is NOT_ADOPTED and B is ADOPT_FOR_FULL_RUN.

**zg6:** G decides only which pools zg6 uses. G6's own verdict (zg6 against zspk, docs/G6_FREE_GNN.md) stays with its
stage. U2d reports zg6 against zspk on both builds beside.

The full run, when a decision sends one, is declared in its own file: six splits per arm under grade-nullx.

## 8. Reported beside, deciding nothing

- FC@5 and hit@1 for every read, as the compares give them;
- on metaqa and webqsp, hit@1 as the headline, with R@5 against its cap. Their R@5 is capped by the metric: many
  questions have more than five golds;
- each carve's ALL, recall, mean rows and both caps, on both builds;
- zg6 against zspk, and zrckp's share of the GNN's gain over RRF, on both builds;
- the base build against P1c3's KT arrays (REPRODUCED or DIFFERS);
- U2d's floors beside the 7 October null's;
- the rows each model scores per question, and the bytes each build's features take.

## 9. What follows (fixed now)

Each verdict closes only the form it tests (docs/SCREENS.md, 3b). The next forms:
- **A NOT_ADOPTED:** the 2 × 2. U2e's rule on today's graph separates the graph from the pool rule. Then via on P1c2's
  pools.
- **B NOT_ADOPTED:** via in the block encoder's own input layer, in place of the heads. Then the link-level form: the
  parent's own first-stage weight and cosine per link.
- **G NOT_ADOPTED:** zg6 with the via columns (zg6v) on the new build.
- In every case: the hit@1 diagnostic on metaqa and webqsp (where the gold sits in the order when hit@1 misses).

## 10. How it runs

**On Modal** (`zu2d_modal.py`):
- **The volume.** A new volume, `mpr-u2d`. U2b's and U2c's volumes are read only.
- **The code.** HEAD's tracked files, byte for byte as the working tree holds them, in one tar per commit. (Line ends
  are the working tree's: core.autocrlf is set, and every script sha256 the records name ignores line ends.) The upload
  refuses while a tracked file differs from HEAD, so every code sha256 a record names is the committed code, and each
  record names its code id (the commit's first 12 characters). Development runs may lay the working tree's changed
  and U2d's own files over HEAD's, as a code id `dev-<commit>-<hash>`; every stage that writes refuses such a code.
- **The inputs.** Inputs not in git are uploaded once, byte for byte, and listed with their sha256 in
  `outputs/u2d/inputs.json` (189 files, 385.5 MB):
  - universal_v2's six-dataset pair (`outputs/universal_v2/six`) and the six base pair's records
    (`outputs/mp_approx_six_base`);
  - the relation tables (`outputs/m3b/relations`);
  - step 1's cache part records of the eleven carves and its two bases (basis_hotpotqa's npz from step 4e's copy, which
    its record's sha256 names);
  - P1c2's seven pools files on the laptop, and step 4d's pools of the eleven carves;
  - the host mirror's manifest (`outputs/host_mirror_six/manifest.json`).

  The probe hashes every input on the volume against the list and files the result. U2c's volume adds what U2c and
  U2e uploaded or wrote (the configs, step 1's carves, step 4d's s1sel pools, P1f's and P1g's stats, the m3b link
  stores, U2b's question files, U2c's and U2e's filed arrays), and U2b's volume the graphs. Each such file is linked in
  one by one; one the code also holds must be the same bytes (up to line ends for text), or the container stops.
- **The package mirror.** The host mirror's selection (`configs/host_mirror_six.yaml`): every file of its manifest is
  linked under `/root/mirror` from U2b's and U2c's volumes, and the few that neither holds (each dataset's small
  top-level files and `graph/`, 1.6 GB) are uploaded to mpr-u2d after hashing against the manifest. `--stage mirror`
  hashes every mirror file against the manifest and files one verify record per group (the package's own files and
  each dataset). Every stage that writes refuses until the inputs and all seven groups are VERIFIED for this manifest.
  `look_u2d.py --modal` and lean_cache read the mirror through look_u2d's `modal_mode` (the package root replaced in
  memory, as host_mode does); open_package's freeze check (58958f33…) still runs.
- **Mounts.** U2b's and U2c's volumes are mounted read only; U2d writes only under mpr-u2d's `/out/<build>` and
  `/out/fits`.
- **Workspaces.** A workspace can reach its spend limit before the stages end. The work then continues in another
  workspace: the files U2d reads or wrote there (the package's files, U2c's, U2e's and U2b's files that U2d links, the
  code, the inputs and every finished item under `/out`) are copied byte for byte through the laptop, under the same
  volume names and paths, and each copy is checked by sha256 against the source's. The probe and `--stage mirror` then
  verify the inputs and the mirror again in the new workspace, and a finished item is reused there by its record, as
  within one workspace. Records name no workspace, and the device (section 5) is the same in every workspace.
- **The images.** The CPU image is zu2e's, with the same pins. The GPU image adds torch 2.8.0 with CUDA.
- **The stages, in order:** upload, probe, mirror, pools (with the check on the new build), then per carve looks,
  caches, chains and links, kt, then copy-bases before them and the gate after, then fits and reads. Each item runs in
  its own container (single use, three retries on an exception; a failed check is not retried), under nonpreemptible
  orchestrators. The pools shards: the new build's sized from U2e's seconds per question on Modal (2 to 42 per carve),
  the base build's half as many, rounded up. An item whose record exists is reused; an interrupted item's files are
  unlinked before it runs again (an overwrite on a volume is discarded at commit).
- **Development runs** (`--stage dev`) come first: the pools and the check (`--limit Q`: the first Q questions, every
  check, nothing written) and the looks (`look_u2d.py --dev Q`: look_x_six's `--limit Q` into a temporary directory,
  every check, then removed). They record every file the process opens (an audit hook), the mirror files read, any
  missing path, the seconds and the peak memory.
- **The outputs.** They go to the volume under `/out/<build>/`, and each fit under `/out/fits/<name>/`. The builds'
  records are fetched to `outputs/u2d/` on the laptop, and each fit's records and reads (never its weights) to
  `outputs/screen/fits/<name>/`, where every screen's fits live (`--out-root` is refused). Records are committed;
  arrays, caches and weights are not.
- **The GPU stages.** `--stage gpu-probe` runs once in each workspace before its fits: the device zu2d.py records,
  whether zu2d.py accepts it, the card's memory, the flags and zu2d.py's self-tests in the GPU image. `--stage fits`
  then runs each fit and its read in one container (zu2d.py train and read in the container's process); a fit whose
  read is stamped is reused, a fit trained but not read is only read, and anything else is unlinked and run again.
- **The calls** (compares, pairs, re-calls and the verdict) run on the laptop, from the fetched reads.

**Size.** The base build's rows equal P1c2's: 53.7 million over the eleven carves. The new build's are about 2.6 times
as many (U2e's pick against P1c2's on s1sel: metaqa 2.17, musique 2.6, hotpotqa 6.0, 2wiki 7.9, webqsp 2.24, squad 1).
On the host, P1c2's roots took 40.3 GB, so the new build's take about 105 GB, about 145 GB on the volume in all.

**ETA**, if nothing fails:
- pools and the check: about 1 hour;
- looks: about 90 job-hours (P1c2's 24.5 and the new build's 65), about 2 hours in parallel;
- caches, builds, KT and the gate: about 2 hours;
- fits and reads: about 24 L40S-hours, about 3 hours in parallel;
- the calls: under an hour.

About 8 to 9 hours in all. The cost is about $80 on Modal: about $30 of CPU and $50 of GPU.

**Self-tests.** zu2d.py is committed in two parts. This commit holds its build part; the fits, reads, null and
decisions follow in the same file, with their self-test, in a commit before the first fit. Passed before this commit:
- `python outputs/mp_unified/zu2d_pools.py --selftest`:
  - both expansions (KIND_T's fill; kt_cos with the budget and the cut) against a direct count;
  - the via columns against a direct computation;
  - the pools file and its added columns;
  - the replacement on both builds;
  - the KT and via columns put into cache order, and their refusals;
  - the rows check by step 1's n sha256;
  - the caps;
  - the shards and the merge;
- `python outputs/mp_unified/zu2d.py --selftest` (the build part):
  - each build's roots in place, and the filed roots restored after, also after an error;
  - the build's refusals (build, what, a chain on a passage dataset);
  - the gate's refusals (missing, FAIL, another build, another declaration);
  - the placements.

Passed before the first fit, in the commit of the second part, its self-test adds:
- the arms (zrckp, zspk and zg6 as filed; zrckpv and zspkv: their widths, their starts bit for bit, their generator);
- the heads and zspkv's propagation by hand, and toy fits through lean_gpu's loop (the head moves; a repeat is the
  same);
- the via carves and their refusals;
- train's and read's refusals (arm, build, fit name, seed, card flags, place, GPU, image, cache root, device, gate);
- one device across the fits of a compare;
- the null from seed fits, and its re-basing; a pair and its re-call under it;
- the four decisions and the outcome table on toy re-calls.

The encoder stays frozen. webqsp never trains. Test splits are never read.

## 11. Results

Not run yet.
