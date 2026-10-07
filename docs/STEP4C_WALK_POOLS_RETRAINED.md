# Step 4c: step 4b's walk pools in the lean MLP, retrained on all six datasets

Declared 7 October 2026, after step 4b's verdict (docs/STEP4B_POOL_REACH_FROZEN_GRAPH.md, Results: ADOPT) and before
any number of this step exists. It is the confirmation that step 4b's section 6 names. Step 4b measured pools only.
This step rebuilds step 1's caches on step 4b's pools, retrains step 1's fits, and grades them on all six datasets,
zero-shot included, under docs/MUSIQUE_DIAGNOSIS.md section 5's rule. Numbers here are development numbers; the
paper's numbers come from one declared confirmation run.

## 1. Question

At each question's frozen pool size, step 4b's walk put all of a question's golds into the pool more often than the
frozen expansion:

- musique: +0.032;
- webqsp: +0.094;
- 2wiki: +0.002;
- metaqa and hotpotqa: AT;
- squad: unchanged, because it has no expansion.

A pool that holds more golds raises what a ranker can reach. It also changes the negatives the ranker trains on.

Do lean MLP fits trained, selected and read on the walk's pools rank better than step 1's fits on the frozen pools, on
every dataset, in-domain and zero-shot?

Everything but the pools is step 1's:

- the carves, the looks' scoring pass and the feature cache;
- the trainer, its config and seed, and the four variants;
- the 36 candidates, the picks and the bootstrap.

A difference is the pools'.

## 2. Pools

**P_F** is step 4b's rule (section 2 of its file):

- personalised PageRank from the frozen seeds over the frozen regime's families, by step 4's local push;
- restart 0.5, the value step 4b chose, and ε = 1e-7;
- order: p descending, then node position.

A question's pool is base ∪ seeds ∪ the first B_q nodes of that order, with B_q = |I_q| − |base_q ∪ seeds_q|. That is
`m3b_compile.build_pool(base, seeds, expansion)` with P_F's expansion in place of the breadth-first one. So every pool
has its frozen pool's size, sorted by node id as the frozen pools are. A question whose walk offers fewer than B_q
nodes keeps the shorter pool and is counted. squad's regime has no family, so its pools are the frozen pools.

**Carves:** the 21 carves step 1 reads, from `outputs/step1/carves.json` (sha256 `53cfb89f…`, checked):

- metaqa, squad, hotpotqa and 2wiki: fit, select (M), s1sel (D) and s1eval;
- musique: s1fit (F'), select, s1sel and s1eval;
- webqsp: s1eval.

The ids of fit and select are M3B's (`m3b_compile.carves_for`). The ids of s1fit, s1sel and s1eval are carves.json's.

**Computation:** the walk needs numba, so `scripts/step4c_walk_pools.py pools` runs on the laptop.

- It imports step 4b's `RegimeUnion` and step 4's kernel unchanged.
- It writes, per carve, each question's id, its frozen pool size, |base ∪ seeds| and P_F's expansion, to
  `outputs/step4c/pools/<dataset>__<carve>.npz`. A manifest, `pools.json`, holds each file's sha256.
- On the s1sel and s1eval carves of step 4b, each question's gold ranks in the recomputed order must equal step 4b's
  filed ranks, or the stage stops.
- Before any walk, each carve's frozen pool sizes are checked against step 1's cache records from the host: every
  part's row total must equal the sum of its questions' frozen pool sizes here, or the stage stops. So the laptop's
  frozen pools are the ones the host's looks build.

## 3. Pipeline

Step 1's (docs/STEP1_MATCHED_SELECTION.md sections 3 to 5). Every stage runs on the host under separate roots, so step
1's files are only read.

- **Looks:** `outputs/mp_unified/look_step4c.py` is look_step1.py with one binding added.
  - After `m3b_compile.prepare` builds a carve's frozen pools, each question's pool is replaced by its P_F pool.
  - The replacement is rebuilt on the host from the host's own base and seeds and the filed expansion. It stops on any
    of these: a missing question; a different frozen pool size or |base ∪ seeds|; an expansion node inside base ∪
    seeds; a pool that is not the filed size.
  - Everything else is look_x_six's scoring pass, unchanged, with `--host --full`: the six pair's scores, the compiled
    columns, the edges and the chunk layout.
  - Output goes to `outputs/step4c/look/<dataset>/<carve>`, with each shard's pools record beside look_x_six's.
- **Cache:** `lean_cache.py` unchanged, `--look-root outputs/step4c/look --out-root outputs/step4c/cache`.
  - Each carve has step 1's part count.
  - The bases are step 1's files, copied and checked against their recorded sha256.
  - lean_cache's own check holds: on every carve, the cache's first 64 rows equal `lean_mlp3.Carve3`'s on the new
    looks.
- **Fits:** `lean_gpu.py train` unchanged, `--cache-root outputs/step4c/cache --out-root outputs/step4c/fits`.
  - The same six fits: J5 and the five leave-outs.
  - The same config `2e-3:1e-4:0.1:8:2`, seed 0, hidden 128 and the variants p, pf, n and nf.
  - The card shares and caps are step 1's.
- **Reads:** `lean_gpu.py read` unchanged on the new caches, over every candidate and step 1's 16 carves per fit.
- **Check:** `lean_gpu.py check` on the new looks: GPU read against CPU Carve3, with step 1's tolerance. Carve3 takes
  the 4c look root through its default argument.
- **Identity gate** (`scripts/step4c_walk_pools.py gate`):
  - squad's pools do not change, so its four carves' cache arrays must equal step 1's, sha256 by sha256. This shows
    that nothing but the pools moved. One array is exempt from bit identity: score2, the six pair's stored scores,
    which the MLP never reads and the grade uses only as references. It must be identical or within 1e-4 (lean_gpu's
    check tolerance) at its largest difference, which is filed.
  - On every other carve, each question's row count must equal its frozen pool size in step 1's cache. A short
    question must instead have |base ∪ seeds| plus its expansion.
  - Every look shard must have filed a pools record naming the manifest's pools file and its counts.
  - If any of these fails, the 4c fits do not start and the grade stops.

## 4. Grading

**Comparator:** step 1's six seed-0 fits on the frozen pools (`outputs/step1/fits`), read on their own s1eval carves.

**Pick:** the rule step 1's verdict adopts.

- If step 1 is ADOPT, it is the D-pick (on the s1sel carves).
- Otherwise it is the M-pick (on the select carves).

Each arm picks on its own select reads. Both picks are reported.

**A read:** one fit on one eval read. Per question with gold:

- R@5 of the 4c fit's pick on the P_F read, minus R@5 of the step-1 fit's pick on the frozen read.
- Ties are broken by pool position, as in step 1.
- The paired bootstrap is step 1's: 2,000 resamples, `default_rng(20261007)`, percentiles 2.5 and 97.5.
- FC@5 and hit@1 are reported beside R@5.

**Labels** on the R@5 interval:

- **ABOVE** if it is above 0;
- **BELOW** if it is below 0;
- **AT** if it holds 0.

**Primary reads:** step 1's eleven:

- J5 on its five training datasets (in-domain);
- each leave-out fit on its held-out dataset (zero-shot);
- J5 on webqsp (zero-shot).

**Verdict:**

- **ADOPT:** no primary read BELOW, and at least one ABOVE. The walk's pools become the pools of later steps.
- **NOT_ADOPTED:** a primary read BELOW. The pools stay as frozen.
- **NO_EFFECT:** every primary read AT.

After ADOPT, seeds 1 and 2 run under the same rules, and the verdict is read again on the three seeds' mean
difference per question.

Secondary reads (each leave-out fit on its four training datasets and on webqsp) get the same labels and are not
graded.

## 5. The early read (reported, not graded)

Step 1's six fits, unchanged, read the six P_F s1eval caches. Their models.pt files are copied to
`outputs/step4c/screen/<fit>` and checked against step 1's train records. Per question, the step-1 pick's R@5 on the
P_F read is compared with the frozen read, with the same pick rule, bootstrap and labels.

This shows whether the pools alone, with the models unchanged, move the ranking. It lands hours before the refit and
decides nothing. The models were trained on the frozen pools' negatives.

## 6. Diagnostics (reported, not graded)

- The primary reads by stratum: metaqa and musique hops, squad answerable, hotpotqa type, 2wiki type. Step 4b moved
  metaqa's two-hop questions up and its three-hop questions down.
- The references on the same rows, P_F against frozen: rrf alone, and the six pair's twin0 and gnn0. The pair's
  scores come from each look, so the frozen GNN and its twin are read on the walk's pools too.
- Each pick's variant and state, the per-variant picks, and every candidate's eval R@5. The eval-best candidate is an
  oracle, never a result.
- The pool counts: questions whose pool changed, short questions, and the share of golds in the pool per read.

## 7. What this does not do

- The encoder and the substrate embeddings are untouched.
- `m3b_pools`, `m3b_compile`, step 4's and step 4b's scripts, look_x_six.py, look_step1.py, lean_cache.py and
  lean_gpu.py are imported unchanged. The bindings above are made in the new files only.
- Step 1's and step 2's files are only read. The served package is read-only.
- No GNN is retrained. The six pair's scores on the new pools are references only.
- webqsp never trains. Test splits and reserved carves are never read.
- Bigger pools (step 4b's frontier) are not tried here. A pool size is a cost choice for a later step.

## 8. Run and ETAs (7 October)

1. Commit this file, `scripts/step4c_walk_pools.py`, `outputs/mp_unified/look_step4c.py` and their tests; push.
2. **Laptop:** the pools of the 21 carves, about 30 minutes, then pushed to the host (about 250 MB).
3. **Host CPU**, idle while the card trains steps 1 and 2:
   - 69 look shards, about 29 shard-hours: 2.5 to 3.5 hours of wall time, about eleven shards at once. The s1eval
     looks go first, for the early read;
   - then the cache parts, step 1's part counts, about 45 minutes;
   - then the identity gate, minutes.
4. **The early read**, as soon as step 1's fits and the 4c eval caches both exist: about 12:30 to 1 pm.
5. **The six 4c fits** on the card, behind step 2's six and only after the identity gate passes. About three fits
   share the card, about 2.5 to 3 hours each. Then their reads and checks.
6. **The grade**, `scripts/step4c_walk_pools.py grade`, once step 1's verdict and the six 4c checks exist. Expected
   around 9 to 11 pm, 7 October.

## Results

None yet.
