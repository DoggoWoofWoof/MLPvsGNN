# GPU task-level qualification

Declaration: [`configs/gpu_task_qualification.yaml`](../configs/gpu_task_qualification.yaml). Record: `outputs/gpu_task_qualification/record.json`. Read 2026-09-29T14:25:29Z at `e0cb6d7`.

## Verdicts

- **host_gpu_det, evaluation: TASK_EQUIVALENT.**
- **host_gpu_det, training: TRAINING_REPRODUCIBLE.**

The question: on fresh probe draws, does the host GPU in the tested determinism mode make the laptop CPU's retrieval decisions for the frozen universal-v2 models, and make them reproducibly? Metric levels on these fit-carve queries are never reported; only differences between placements are.

## Probe

12 batches (stage-2 seed-0 batches 11–22), 192 query draws (2wiki 32, hotpotqa 33, metaqa 32, musique 31, squad 29, webqsp 35), 0.994 GB. None of them is a draw of the equivalence probe.

## The candidate against the laptop reference

Q_DET (repeat bit-identical): True. Score cells hold on both runs: False. Ranking or metric differences away from a near tie, both runs: 0.

| model | score cells failing | largest ratio | top-1 differs | top-5 set differs | excused (near tie) | ranking or metric differences |
|---|---:|---:|---:|---:|---:|---:|
| `u_gnn_v2_ef__H128__six__s0` | 0 | 0.56 | 0 | 0 | 0 | 0 |
| `u_gnn_v2_ef__H128__six__s1` | 0 | 0.83 | 0 | 0 | 0 | 0 |
| `u_gnn_v2_ef__H128__six__s2` | 0 | 0.65 | 0 | 0 | 0 | 0 |
| `u_mlp_v2_mix__H128__s0` | 0 | 0.73 | 0 | 0 | 0 | 0 |
| `u_mlp_v2_mix__H128__s1` | 2 | 1.36 | 0 | 0 | 0 | 0 |
| `u_mlp_v2_mix__H128__s2` | 0 | 0.94 | 0 | 0 | 0 | 0 |

Task-metric differences (candidate minus reference; the per-dataset means of recall@5, full_coverage@5 and hit@1):

| model | draws with a task metric differing | largest |Δ| of a dataset mean (R@5, FullCov@5, Hit@1) |
|---|---:|---:|
| `u_gnn_v2_ef__H128__six__s0` | 0 | 0 |
| `u_gnn_v2_ef__H128__six__s1` | 0 | 0 |
| `u_gnn_v2_ef__H128__six__s2` | 0 | 0 |
| `u_mlp_v2_mix__H128__s0` | 0 | 0 |
| `u_mlp_v2_mix__H128__s1` | 0 | 0 |
| `u_mlp_v2_mix__H128__s2` | 0 | 0 |

## The floor (laptop, 4 threads against 8)

| model | score cells failing | largest ratio | top-1 differs | top-5 set differs | excused (near tie) | ranking or metric differences |
|---|---:|---:|---:|---:|---:|---:|
| `u_gnn_v2_ef__H128__six__s0` | 0 | 0.52 | 0 | 0 | 0 | 0 |
| `u_gnn_v2_ef__H128__six__s1` | 0 | 0.57 | 0 | 0 | 0 | 0 |
| `u_gnn_v2_ef__H128__six__s2` | 0 | 0.55 | 0 | 0 | 0 | 0 |
| `u_mlp_v2_mix__H128__s0` | 0 | 0.74 | 0 | 0 | 0 | 0 |
| `u_mlp_v2_mix__H128__s1` | 0 | 0.73 | 0 | 0 | 0 | 0 |
| `u_mlp_v2_mix__H128__s2` | 0 | 0.71 | 0 | 0 | 0 | 0 |

## Training reproducibility (T5)

| model | step losses equal | final state sha256 equal | final scores equal |
|---|---|---|---|
| `u_gnn_v2_ef__H128__six__s0` | True | True | True |
| `u_mlp_v2_mix__H128__s0` | True | True | True |

## Beside the verdicts (never gating)

### host CPU (8 threads) against the laptop reference

Score cells failing: 0; ranking or metric differences away from a near tie: 0; excused near-tie draws: 0; raw bit-identical: True.

| model | score cells failing | largest ratio | top-1 differs | top-5 set differs | excused (near tie) | ranking or metric differences |
|---|---:|---:|---:|---:|---:|---:|
| `u_gnn_v2_ef__H128__six__s0` | 0 | 0.00 | 0 | 0 | 0 | 0 |
| `u_gnn_v2_ef__H128__six__s1` | 0 | 0.00 | 0 | 0 | 0 | 0 |
| `u_gnn_v2_ef__H128__six__s2` | 0 | 0.00 | 0 | 0 | 0 | 0 |
| `u_mlp_v2_mix__H128__s0` | 0 | 0.00 | 0 | 0 | 0 | 0 |
| `u_mlp_v2_mix__H128__s1` | 0 | 0.00 | 0 | 0 | 0 | 0 |
| `u_mlp_v2_mix__H128__s2` | 0 | 0.00 | 0 | 0 | 0 | 0 |

### host GPU against host CPU, one machine

Score cells failing: 2; ranking or metric differences away from a near tie: 0; excused near-tie draws: 0; raw bit-identical: False.

| model | score cells failing | largest ratio | top-1 differs | top-5 set differs | excused (near tie) | ranking or metric differences |
|---|---:|---:|---:|---:|---:|---:|
| `u_gnn_v2_ef__H128__six__s0` | 0 | 0.56 | 0 | 0 | 0 | 0 |
| `u_gnn_v2_ef__H128__six__s1` | 0 | 0.83 | 0 | 0 | 0 | 0 |
| `u_gnn_v2_ef__H128__six__s2` | 0 | 0.65 | 0 | 0 | 0 | 0 |
| `u_mlp_v2_mix__H128__s0` | 0 | 0.73 | 0 | 0 | 0 | 0 |
| `u_mlp_v2_mix__H128__s1` | 2 | 1.36 | 0 | 0 | 0 | 0 |
| `u_mlp_v2_mix__H128__s2` | 0 | 0.94 | 0 | 0 | 0 | 0 |

### host GPU with TF32 on (diagnosis) against the laptop reference

Score cells failing: 1152; ranking or metric differences away from a near tie: 1; excused near-tie draws: 0; raw bit-identical: False.

| model | score cells failing | largest ratio | top-1 differs | top-5 set differs | excused (near tie) | ranking or metric differences |
|---|---:|---:|---:|---:|---:|---:|
| `u_gnn_v2_ef__H128__six__s0` | 192 | 391.74 | 0 | 0 | 0 | 0 |
| `u_gnn_v2_ef__H128__six__s1` | 192 | 536.54 | 0 | 0 | 0 | 0 |
| `u_gnn_v2_ef__H128__six__s2` | 192 | 439.35 | 0 | 0 | 0 | 0 |
| `u_mlp_v2_mix__H128__s0` | 192 | 530.56 | 0 | 0 | 0 | 0 |
| `u_mlp_v2_mix__H128__s1` | 192 | 714.09 | 0 | 0 | 0 | 0 |
| `u_mlp_v2_mix__H128__s2` | 192 | 609.73 | 0 | 1 | 0 | 1 |

## Placements and inventory

| arm | host | env | torch | device | TF32 | deterministic | threads | seconds |
|---|---|---|---|---|---|---|---:|---:|
| laptop_cpu_t8 | Inspiron-14 | Python313 | 2.8.0+cpu | cpu | – | False | 8 | 167.1 |
| laptop_cpu_t4 | Inspiron-14 | Python313 | 2.8.0+cpu | cpu | – | False | 4 | 249.7 |
| host_gpu_det | DESKTOP-SLQMEQH | mpr-cu128@62fc45e9e1ba | 2.8.0+cu128 | NVIDIA RTX 4500 Ada Generation | (False, False) | True | 8 | 12.7 |
| host_gpu_det_r2 | DESKTOP-SLQMEQH | mpr-cu128@62fc45e9e1ba | 2.8.0+cu128 | NVIDIA RTX 4500 Ada Generation | (False, False) | True | 8 | 11.7 |
| host_cpu_t8 | DESKTOP-SLQMEQH | mpr-cpu@31803e6457ab | 2.8.0+cpu | cpu | – | False | 8 | 43.6 |
| host_gpu_tf32 | DESKTOP-SLQMEQH | mpr-cu128@62fc45e9e1ba | 2.8.0+cu128 | NVIDIA RTX 4500 Ada Generation | (True, True) | True | 8 | 6.8 |

Inventory per arm (versions, libraries, autocast): `record.json`, `arms.<arm>.inventory`. Autocast was off in every arm (an arm with autocast on is a hard stop).

## What this opens

Read `configs/gpu_task_qualification.yaml#what_a_verdict_opens` and `#host_native_protocol`. A verdict opens no stage by itself: a later stage names the placement in a dated authorization block that quotes the tested settings verbatim and runs mirror_verification. A GPU fit is a new draw, never paired with a laptop seed.
