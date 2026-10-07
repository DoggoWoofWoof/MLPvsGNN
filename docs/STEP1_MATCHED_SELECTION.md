# Step 1: distribution-matched selection, on all six datasets

Declared 7 October 2026, before any number of this step exists. It is step 1 of docs/MUSIQUE_DIAGNOSIS.md section 5,
under that section's rule: a change to the training of every dataset, graded on every dataset, zero-shot included.
Numbers here are development numbers; the paper's numbers come from one declared confirmation run.

## 1. Question

Each fit picks its epoch and configuration on a select carve. M3B's select carve is a stride over each dataset's
train split, so it can share gold passages with the fit carve where the eval population shares none: on musique 0.945
of its questions hold a fit gold and 0.941 share a single-hop component with a fit question, against 0 for dev. The
pick then rewards remembering training golds, which no eval rewards (MUSIQUE_DIAGNOSIS section 3).

Does picking on a select carve matched to each dataset's eval population (D) give a model that reads at least as well
as picking on M3B's select carve (M) on every dataset, and better on some, in-domain and zero-shot?

Both picks come from the same training runs. The step changes which saved state is chosen, nothing else, so a
difference between the arms is the selection's alone.

## 2. Carves (filed)

`scripts/step1_carves.py` wrote `outputs/step1/carves.json`, sha256
`53cfb89f1e41a76257113719639f96318d14279f86b50a970f9a0bbf6686d746` (script sha256 `2a1cf642…`, freeze
`58958f33…`). It lists question ids only; no score, weight or model was read. The rules are in its docstring:

- **cell** of a question: (o, t). o = 1 if one of its gold nodes is a gold node of a fit question; t = its stratum
  (metaqa hop, squad answerable, musique hop, hotpotqa level/type, 2wiki type).
- **target**: the eval population's share of each cell. **size**: M3B's select carve's size, so the pooled select
  weighs the datasets as before. **quotas**: largest remainder.
- **candidates**: train questions outside the fit carve and the reserved carves (2wiki x2 and x3, musique x3),
  zero-gold out, taken in sha256 order per cell.
- **musique**: questions linked through a shared single-hop component form 1,012 groups. Whole groups go to the select
  side (134 groups, 3,936 questions), so the matched select carve shares no component with the fit. The fit is
  regrouped (F', 4,601 questions) with M3B's fit's hop counts (3,318 / 1,012 / 271), from the other groups, outside x3
  and outside M3B's select carve. Both arms train on F'; M's select carve is M3B's, as before.
- **eval reads**: each dataset's M3B eval population; metaqa reads every fourth of its 39,138 dev questions.

Every quota filled; no cell was short.

| dataset | fit | M select (holds a fit gold) | D select (holds a fit gold) | eval read (holds a fit gold) |
| --- | --- | --- | --- | --- |
| metaqa | 5,960 | 1,497 (0.828) | 1,497 (0.864) | 9,785 dev (0.864) |
| squad | 5,856 | 1,498 (0.290) | 1,498 (0) | 11,873 dev (0) |
| musique | 4,601 (F') | 1,534 (0.757) | 1,534 (0) | 2,417 dev (0) |
| hotpotqa | 5,930 | 1,508 (0.163) | 1,508 (0.182) | 7,405 validation (0.183) |
| 2wiki | 5,928 | 1,496 (0.429) | 1,496 (0.308) | 12,576 dev (0.308) |
| webqsp | never trains | - | - | 1,503 train_holdout |

These are shares of question ids, not results. Two more shifts the strata carry: hotpotqa's validation questions are
all `hard` while its M select carve is 15% `hard`, and musique's dev mix is 52 / 31 / 17% two / three / four hops
against the M select carve's 72 / 22 / 6%. The D carves match the eval mix to the third decimal.
M's fit and select digests equal the filed looks' `carve_ids_sha256` on all five training datasets.

## 3. Looks

`outputs/mp_unified/look_step1.py` runs look_x_six's scoring pass unchanged (`--host --full`) on the new carves:
`s1sel` (five datasets), `s1fit` (musique's F') and `s1eval` (six datasets). It checks carves.json against the sha256
above and each carve's ids against its recorded sha256; an eval read takes its rows, cache rows and golds from
`m3b_compile.population`'s eval population, checked against the recorded digest. M's fit and select carves reuse the
filed looks. 37 shards, about 15 shard-hours on the host.

## 4. Model, feature cache and trainer

**Model**: lean_mlp8's `LeanMLP8` (lean_mlp7g's lean MLP with the nan-safe z-score), non-MP: learned weights over fixed
per-candidate columns. Hidden 128, config `2e-3:1e-4:0.1:8:2` (lr, weight decay, dropout, 8 epochs, SWA from epoch
2), seed 0. Four variants, the candidates of the pick:

| variant | block set | context |
| --- | --- | --- |
| p | pick = rank + dense_cos + topo_STRUCT + depth_STRUCT + SEMB + SEED + WALK + DISTS + WALKF | none |
| pf | pick | film |
| n | pns = pick without SEMB | none |
| nf | pns | film |

**Store**: pca256. The basis is fitted once per basis graph on the host (`lean_cache.py --make-basis`, 60,000 nodes,
seed 0, two BLAS threads) and pinned by sha256 in every cache record. A fit's basis is the first of 2wiki, hotpotqa
that it trains on: 2wiki for every fit but leave-out-2wiki, which uses hotpotqa's, so no zero-shot read uses a basis
fitted on its own graph.

**Feature cache** (`outputs/mp_unified/lean_cache.py`): per carve, the arrays lean_mlp8's batches read for these
blocks, computed once on the host by the same per-query functions lean_mlp, lean_mlp2 and lean_mlp3 call: the compiled
columns of rank, dense_cos, topo_STRUCT and depth_STRUCT, WALK, WALKF, gold flags and gold totals, the float16 query
embeddings, the pool node ids with the int8 store codes, and SEED and DISTS on each basis. Check: on every carve the
cache's first 64 rows equal `lean_mlp3.Carve3`'s (built with `limit=64` in the same process) bit for bit, block by
block, codes included; a node whose codes differ between two chunks keeps per-row codes.

**Trainer** (`outputs/mp_unified/lean_gpu.py`): fit8's loop for arm ctl, on the cache, on the GPU: the same model
class and initial weights, the same unit order, chunks of 32 split by carve, Adam, dropout, the step guard and SWA,
with `torch.use_deterministic_algorithms(True)` and TF32 off. Its segment operations are lean_mlp's, written for any
device. It saves the eight epoch states and the SWA state. Checks before any fit is read:

- **CPU identity**: on the CPU, the trainer's states after two epochs equal fit8's on the same carves bit for bit
  (2wiki fit and select, run on the laptop).
- **Read identity**: for each fit and each carve it reads, every candidate's scores on the carve's first 64 rows from
  `lean_mlp.scores_of` on a `Carve3` (CPU) and from the trainer's read (GPU). The GPU is not bit-identical to the
  CPU; the check fails if a score differs by more than 1e-4, or a row's R@5 differs where its fifth and sixth
  scores are more than 1e-4 apart.

## 5. Fits and candidates

Six fits on seed 0:

| fit | trains on (fit carves) | basis | zero-shot reads |
| --- | --- | --- | --- |
| J5 | metaqa, squad, musique (F'), hotpotqa, 2wiki | 2wiki | webqsp |
| L-metaqa | the other four | 2wiki | metaqa, webqsp |
| L-squad | the other four | 2wiki | squad, webqsp |
| L-musique | the other four | 2wiki | musique, webqsp |
| L-hotpotqa | the other four | 2wiki | hotpotqa, webqsp |
| L-2wiki | the other four | hotpotqa | 2wiki, webqsp |

A fit's candidates are its four variants' nine saved states (epochs 0 to 7 and SWA): 36. Every candidate reads the
fit's M select carves, its D select carves and all six eval reads; per-row R@5, FC@5 and hit@1 are saved (ties broken
by pool position, as the look's metrics).

## 6. Picks

- **M-pick**: the candidate with the highest mean, over the fit's pooled M select rows, of (R@5 + FC@5) / 2 (fit8's
  rule, over variants as well as states).
- **D-pick**: the same on the fit's pooled D select rows.
- A tie goes to the candidate first in the order p, pf, n, nf, then epoch 0 to 7, then SWA.

Per-variant picks (the state only) are reported as well, not graded.

## 7. Grading

A read is a fit's two picks on one eval read: per question, D-pick minus M-pick; the paired bootstrap over questions
(2,000 resamples, seed 20261007, percentile 95% interval).

- **SAME**: both picks are the same candidate (the difference is 0 by construction).
- **ABOVE** / **BELOW**: the interval is above / below 0. **AT**: it holds 0.

**Primary reads** (eleven): J5 on its five training datasets (in-domain), each leave-out fit on its held-out dataset
(zero-shot), and J5 on webqsp (zero-shot). Graded on R@5; FC@5 and hit@1 reported beside it.

**Verdict** (seed 0):

- **ADOPT**: no primary read BELOW, and at least one ABOVE.
- **NOT_ADOPTED**: a primary read BELOW. A loss on one dataset blocks adoption.
- **NO_EFFECT**: every primary read SAME or AT.

Secondary reads (each leave-out fit on its four training datasets, the leave-out fits on webqsp) are reported with the
same labels and are not graded.

After ADOPT, seeds 1 and 2 run under the same rules and the verdict is read again on the three seeds' mean difference
per question. After ADOPT, later training steps pick on the matched select carves. In every case, later steps save every
epoch state and report both picks, which costs a few reads.

## 8. Diagnostics (reported, not graded)

- Per dataset, the rank correlation (Spearman) between the 36 candidates' select quality and their eval quality, for
  the M and the D carves: which select carve predicts the eval.
- Every candidate's eval quality, and the eval-best candidate. The eval-best is an oracle; it is never a result.
- The primary reads by stratum: metaqa and musique hops, squad answerable, hotpotqa type, 2wiki type.
- Each pick's variant and state, and the per-variant picks.

## 9. Order of work and ETAs (7 October)

1. Commit this file, `scripts/step1_carves.py`, `outputs/step1/carves.json` and `look_step1.py`; push.
2. On the host: the two bases (minutes); caches of the filed carves (M fits and selects) right away; the 37 look
   shards through the feeder (about 15 shard-hours; mpr is near its memory cap, so 3 to 5 hours of wall time).
3. Caches of the new carves as their looks finish (about an hour of CPU in parallel parts).
4. Six GPU fits (four variants each; about 3 minutes per variant on the GPU against 5 to 55 hours on two CPU threads
   for the earlier joint lean fits), then their reads, two runs sharing the card.
5. The grade (`scripts/step1_grade.py`), and the results below. Expected around midday, 7 October.

Revised 03:05, 7 October (no number of this step exists yet). The look shards stopped three times within seconds on
inputs the host no longer held: the six base's stage-2 checkpoints, the stage-2 eval arrays and webqsp's relation
embeddings. Each was pushed back from the laptop and checked against its pinned sha256. The shards now wait on two
two-question smokes on the host, which passed, and they started at 02:46. The GPU trainer's CPU identity with fit8 is
IDENTICAL on all four variants; a 2wiki smoke on the card (each run twice; a repeat that differs drops the fits)
and its CPU read check gate the six fits and their reads. The grade runs on the host once all six checks pass. Expected
between 9 and 11 am, 7 October.

Revised 04:30, 7 October (no number of this step exists yet). The smoke passed on the card at 03:45: both repeats
IDENTICAL, the read check PASS. A training step takes 65 to 130 ms beside another GPU run, so a variant is about 43
minutes (8 epochs of about 320 s), not 3. A fit (four variants) is about 2.8 hours. The fits need a fifth to a third
of the card (process peaks 3.8 to 6.3 GB), so J5 and the heavy leave-outs take 0.3 of it and L-metaqa and L-musique
0.2: four fits train at once. L-musique, sent at half the card, was restarted at 0.2 as s1i-train-L-musique.
Expected: the first four fits by about 7:45 am, the other two by about 10:30, the grade about 11:00 to 11:30 am.

Revised 05:15, 7 October (no number of this step exists yet). A fit's GPU peak is the context-statistics pass at the
start of each FiLM variant, not its training epochs. That pass is float64 over 256-question batches: about 520k
rows on metaqa and musique, about 1.3 GB above the resident carves. Both L-musique fits ran out of memory there
under the 0.18 cap, after their first variant, and restarted from scratch with no cap (retry_cmd). The first
epoch's loss repeated exactly. The unsent fits now cap at 0.32 of the card (heavy; share 0.3) and 0.22 (L-metaqa;
share 0.2). Expected: the last fit (L-2wiki) starts about 8:05 am, and the grade lands about 11:30 am.

Revised 06:15, 7 October (no number of this step exists yet). Three things happened between 05:36 and 06:05:

- Two metaqa eval cache shards failed on a transient Windows file lock. The feeder then dropped every read,
  check and grade of both steps. The shards were rebuilt under retry_cmd. The reads now wait on a cache gate
  (outputs/host_ops/wait_cache.py) instead of on each shard, and were renamed s1j-* (step 1) and s2x-* (step 2).
  The gate passed at 05:46.
- The card ran out of memory and spilled into system memory: 23.4 of 24 GB used, 1.0 GB shared. The three fits
  rerun with no cap kept their peaks. Four fits plus their CUDA contexts no longer fit, and every fit ran about
  ten times slower. Step 2's L-musique fit was cancelled at 05:57 and requeued with a cap (s2y-train-L-musique).
  The card dropped to 17.0 GB and the epochs went back to normal (L-metaqa 189 s).
- J5 ran out of memory at its context pass under 6.72 GiB (5.62 GiB allocated, 0.90 GiB fragmented, a 344 MiB
  request). It restarted from scratch with no cap at 06:04. The unsent heavy fits are now capped at 0.31 of the
  card (7.44 GiB), with a share of 0.33 that counts the CUDA context. So at most three heavy fits share the card.

Expected: L-squad starts about 7:15 am, L-hotpotqa about 7:30 and L-2wiki about 8:45 (after J5). The last fit ends
about 10:50, and the grade lands about 11:30 am to noon.

## 10. What this does not do

- The encoder and the substrate embeddings stay frozen; no step will change them.
- The fit carves are M3B's, except musique's regroup, which both arms share.
- No GNN here: the QD-GNN confirmation is a later declared file, after the seed-0 verdict.
- webqsp never trains. Test splits are never read.
- Steps 2 to 4 (a difficulty-balanced fit, depth, pool reach) are not run here; each needs its own file.

## Results

None yet.
