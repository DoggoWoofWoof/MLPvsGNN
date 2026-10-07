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

Development numbers; the paper's numbers come from one declared confirmation run.

### Pools and identity gate (7 October, before any fit)

- **Pools:** the laptop built the pools of the 21 carves. There are 88,900 questions; 52,512 pools changed and none is
  short. The manifest `pools.json` has sha256 `c7af9bac…`.
- **Looks and cache:** the 69 look shards and the cache parts ran on the host.
- **Gate:** `outputs/step4c/gate.json` (05:44 UTC, script sha256 `ee5398f4…`) is PASS.
  - squad's four carves have step 1's cache arrays sha256 for sha256, score2 included (largest difference 0).
  - On every other carve, step 1's rows are the frozen pool sizes and the new rows are P_F's; no question is short.
  - Every look filed its pools record.

### Early read (12:33; reported, not graded)

Step 1's six fits, unchanged, were read on the P_F s1eval caches (`outputs/step4c/screen.md`). Each cell is R@5 on P_F
minus the same pick on the frozen pools, with the M-pick that step 1's verdict adopts.

| fit | read | role | R@5 frozen | R@5 P_F | diff [95%] | label |
| --- | --- | --- | --- | --- | --- | --- |
| J5 | metaqa | in-domain | 0.6443 | 0.6366 | −0.0077 [−0.0106, −0.0049] | BELOW |
| J5 | squad | in-domain | 0.9095 | 0.9095 | 0 (no pool changed) | AT |
| J5 | musique | in-domain | 0.5609 | 0.5525 | −0.0083 [−0.0123, −0.0043] | BELOW |
| J5 | hotpotqa | in-domain | 0.9013 | 0.9022 | +0.0009 [−0.0003, +0.0021] | AT |
| J5 | 2wiki | in-domain | 0.8694 | 0.8698 | +0.0003 [−0.0005, +0.0011] | AT |
| J5 | webqsp | zero-shot | 0.1167 | 0.1155 | −0.0013 [−0.0092, +0.0061] | AT |
| L-metaqa | metaqa | zero-shot | 0.0888 | 0.0945 | +0.0058 [+0.0037, +0.0080] | ABOVE |
| L-squad | squad | zero-shot | 0.8797 | 0.8797 | 0 (no pool changed) | AT |
| L-musique | musique | zero-shot | 0.2696 | 0.2882 | +0.0186 [+0.0131, +0.0244] | ABOVE |
| L-hotpotqa | hotpotqa | zero-shot | 0.8510 | 0.8502 | −0.0009 [−0.0022, +0.0004] | AT |
| L-2wiki | 2wiki | zero-shot | 0.8070 | 0.8076 | +0.0005 [−0.0005, +0.0015] | AT |

- **Unchanged models, new pools.** The models trained on metaqa and musique lose there. On metaqa, J5 and every
  leave-out fit that trains on it are BELOW (−0.0074 to −0.0081). On musique, J5 (−0.0083) and two of the four
  leave-out fits that train on it (L-metaqa −0.0056, L-squad −0.0094) are BELOW; the other two are AT. The fits that
  never saw metaqa or musique gain on them: metaqa +0.0058 and musique +0.0186. webqsp is mixed: L-musique −0.0118
  (BELOW), L-2wiki +0.0083 (ABOVE), the others AT. hotpotqa and 2wiki are AT throughout.
- **The pools themselves** (eval reads, frozen → P_F):

  | dataset | every gold in the pool | golds in the pool |
  | --- | --- | --- |
  | metaqa | 0.8949 → 0.8980 | 0.9565 → 0.9467 |
  | musique | 0.8026 → 0.8349 | 0.9207 → 0.9339 |
  | hotpotqa | 0.9684 → 0.9695 | 0.9800 → 0.9805 |
  | 2wiki | 0.9059 → 0.9079 | 0.9637 → 0.9645 |
  | webqsp | 0.6880 → 0.7824 | 0.8110 → 0.8834 |
  | squad | 0.9798 (unchanged) | 0.9798 (unchanged) |

  On metaqa, the walk puts every gold in the pool slightly more often but holds fewer golds overall (−0.0098). R@5
  counts golds, and no model ranks a gold its pool does not hold, so J5's metaqa loss (−0.0077) can stay after the
  refit. On musique the pools hold more golds (+0.0132), so the loss of the unchanged models there is theirs (they
  trained on the frozen pools' negatives); the refit tests exactly that.
- **References on the same rows** (P_F − frozen): the twin on metaqa −0.0035 (BELOW) and the GNN +0.0029 (ABOVE); on
  musique the GNN −0.0080 (BELOW) and the twin AT; on webqsp the GNN +0.0149 (ABOVE) and the twin +0.0099 (AT); every
  other reference AT or unchanged.

### Grade

Held at about 14:00, 7 October, under the user's rule of one training run per idea (docs/SCREENS.md). The six 4c
fits and the grade do not run. One screen fit replaces them: scr-pools, step 1's L-musique split trained on the 4c
cache. It is compared with step 1's L-musique (this decides) and with this early read. If the screen shows a gain,
the six fits and the grade run as this file declares them.

**Screen result (15:12): MIXED, so the six fits and the grade do not run** (`outputs/screen/scr-pools.md`). Against step
1's L-musique, musique read zero-shot gains (+0.0203 R@5) and webqsp read zero-shot loses (−0.0200). The other four
reads are within. These are development numbers.

Before the hold: The six 4c fits queue behind step 2's six fits and reads. Step 2's fits started late (J5 at 11:00, the others
from 12:21 on, once step 1's reads and this early read had freed the card). Expected: the 4c fits from about 4 pm, two
waves of three at once, and the grade about 11 pm to 1 am, 7 to 8 October.
