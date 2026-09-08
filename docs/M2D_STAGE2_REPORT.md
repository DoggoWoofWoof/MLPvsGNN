# M2D Stage 2 - report

**Verdict: STOP_S4_DEVELOPMENT_CONFIRMED**

musique_clean/R1 does not reach -0.50pp on the three-seed mean. The guard is section 12's and applies to both blockers.

Applied by `scripts/m2d_stage2_gate.py`, whose rule and threshold were committed before any Stage-2 fit existed -- which is why that gate's own tests run on synthetic fits. Authorised by stage_2_amendment, filed 2026-09-08 by review. 4 of 4 new fits present; 6 of 6 same-seed rows.

Four new fits at `32bc220a35ee`: A3_MINIMAL alone, at seeds 1 and 2, on squad_clean/R1 and musique_clean/R1. A3_MINIMAL at seed n against S3 at seed n, both blockers, seeds 0, 1 and 2. Seed 0 is Stage 1's own artifact and every S3 and S4 row is M2B's; nothing already fit was refit to produce a comparison row. No test data was read: every fit reports `test_split_read: false`.

The reason these seeds were run is section 15b's, not a paraphrase of it. Seed 0 landed inside a **pre-measured decision-uncertainty band**: A3_MINIMAL's MuSiQue miss to the admissibility guard was 0.065pp while that cell's already-measured seed variation was 0.816pp, an order of magnitude wider. Review therefore authorised the minimum additional seeds needed to resolve the scientific decision -- two -- with the aggregation fixed in advance as the arithmetic mean of the three same-seed differences, so that no seed could be chosen after the fact.

## Per-seed metrics (section 15b)

Each seed's A3-MINIMAL against **its own** S3 and S4 rows. Rows marked `*` were reused rather than newly fit: every S3 and S4 row is M2B's at that seed, and seed 0's arm row is Stage 1's own artifact.

### squad_clean/R1

| seed | model | R@1 | R@5 | R@20 | MRR | A3-S3 R@5 pp | A3-S4 R@5 pp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | S3 * | 0.732400 | 0.917706 | 0.967197 | 0.813783 | - | - |
| 0 | S4 * | 0.684635 | 0.910224 | 0.963936 | 0.784993 | - | - |
| 0 | **A3_MINIMAL** * | 0.726453 | 0.918473 | 0.967197 | 0.810455 | **+0.077** | +0.825 |
| 1 | S3 * | 0.727796 | 0.916747 | 0.966238 | 0.810622 | - | - |
| 1 | S4 * | 0.678688 | 0.906772 | 0.963744 | 0.781293 | - | - |
| 1 | **A3_MINIMAL** | 0.726837 | 0.915787 | 0.966046 | 0.810126 | **-0.096** | +0.902 |
| 2 | S3 * | 0.728371 | 0.919049 | 0.967006 | 0.811855 | - | - |
| 2 | S4 * | 0.696144 | 0.910033 | 0.964895 | 0.791514 | - | - |
| 2 | **A3_MINIMAL** | 0.728180 | 0.915979 | 0.965663 | 0.811147 | **-0.307** | +0.595 |

### musique_clean/R1

| seed | model | R@1 | R@5 | R@20 | MRR | A3-S3 R@5 pp | A3-S4 R@5 pp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | S3 * | 0.444166 | 0.773526 | 0.873275 | 0.928313 | - | - |
| 0 | S4 * | 0.311167 | 0.750941 | 0.867629 | 0.771464 | - | - |
| 0 | **A3_MINIMAL** * | 0.397114 | 0.767880 | 0.871393 | 0.873413 | **-0.565** | +1.694 |
| 1 | S3 * | 0.444166 | 0.772898 | 0.876412 | 0.928263 | - | - |
| 1 | S4 * | 0.322459 | 0.748432 | 0.865747 | 0.780834 | - | - |
| 1 | **A3_MINIMAL** | 0.398369 | 0.760979 | 0.868256 | 0.874406 | **-1.192** | +1.255 |
| 2 | S3 * | 0.442911 | 0.772898 | 0.878294 | 0.926787 | - | - |
| 2 | S4 * | 0.322459 | 0.742785 | 0.861355 | 0.783396 | - | - |
| 2 | **A3_MINIMAL** | 0.415935 | 0.757842 | 0.865119 | 0.894080 | **-1.506** | +1.506 |

## The three-seed decision (section 15b)

The filed criterion: the same-seed `A3_MINIMAL - S3` recall@5 difference, aggregated as the arithmetic mean of the three same-seed differences, must reach **-0.50pp** on **both** blockers. The threshold is section 12's, unchanged; what section 15b fixed was the aggregation, and it fixed it before these fits existed.

| blocker | seed 0 | seed 1 | seed 2 | mean | sample SD | signs | meets bound |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| squad_clean/R1 | +0.077 | -0.096 | -0.307 | **-0.109** | 0.192 | +-- | **yes** |
| musique_clean/R1 | -0.565 | -1.192 | -1.506 | **-1.087** | 0.479 | --- | **no** |

The per-seed signs and the sample SD are reported beside the mean and do not enter the decision. Section 15b: the declaration's existing convention reports `negative_in_every_seed` beside its three-seed means rather than gating on it, and no per-seed floor was ever filed.

## Robust top-rank behaviour (section 15b)

A3_MINIMAL was selected to repair top-rank ordering, so rank-1 and MRR are the mechanism-specific evidence rather than supporting detail. Three-seed means of the same-seed differences, in percentage points.

| blocker | metric | mean A3-S3 pp | mean A3-S4 pp |
| --- | --- | ---: | ---: |
| squad_clean/R1 | recall@1 | -0.237 | +4.067 |
| squad_clean/R1 | recall@5 | -0.109 | +0.774 |
| squad_clean/R1 | recall@20 | -0.051 | +0.211 |
| squad_clean/R1 | mrr | -0.151 | +2.464 |
| musique_clean/R1 | recall@1 | -3.994 | +8.511 |
| musique_clean/R1 | recall@5 | -1.087 | +1.485 |
| musique_clean/R1 | recall@20 | -0.774 | +0.335 |
| musique_clean/R1 | mrr | -4.715 | +10.207 |

Both columns are true at once, and the second is the substance of what was learned. Against **native S4** the added column helps everywhere, at every seed, on both blockers, and most of all exactly where it was predicted to -- at rank 1. Against **S3** it does not close the gap. `semantic_difference` repairs a real part of what the S4 widening cost; it does not repair enough of it, and the guard is against S3.

## Identity verification (section 15b)

What a training seed can attest to is the arm it actually built. Section 15b does not rerun the systems benchmark per seed -- parameter count and operation graph do not depend on the seed -- so **no new latency result is claimed here** and Stage 1's p95 figures stand as Stage 1's.

| blocker | seed | semantic | scorer | total | added semantic | precomputed |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| squad_clean/R1 | 1 | 198,144 | 8,641 | 206,785 | 1,536 | false |
| squad_clean/R1 | 2 | 198,144 | 8,641 | 206,785 | 1,536 | false |
| musique_clean/R1 | 1 | 198,144 | 8,641 | 206,785 | 1,536 | false |
| musique_clean/R1 | 2 | 198,144 | 8,641 | 206,785 | 1,536 | false |

Identity holds on every new fit: **true**. The added column is exactly the 1,536 semantic parameters section 3 authorised, nothing is cached or precomputed between queries, and all four containers report one timed path (1 distinct description): "the whole uncached forward: the structural columns, all of native S4's projection work, this arm's added column and its reduction, and the widened scorer. Section 10's primary number, cold, with nothing precomputed between queries."

## Provenance and panel identity

| blocker | seed | run id | source commit | rows | panel digest |
| --- | ---: | --- | --- | ---: | --- |
| squad_clean/R1 | 0 * | `fc-01M1ZNZVAZEQ1YN2GQV8MFQGHC` | `0e498cf0d392` | 5,213 | `a5e626f89b54e574` |
| squad_clean/R1 | 1 | `fc-01M1ZY6AX754QEWXE8G24FEA09` | `32bc220a35ee` | 5,213 | `a5e626f89b54e574` |
| squad_clean/R1 | 2 | `fc-01M1ZY6D3ZJ0K69YG3SMN9NFCA` | `32bc220a35ee` | 5,213 | `a5e626f89b54e574` |
| musique_clean/R1 | 0 * | `fc-01M1ZNZY32SZ8BHWJH59RMSP8S` | `0e498cf0d392` | 797 | `0effc543c750a331` |
| musique_clean/R1 | 1 | `fc-01M1ZY6F8G5QZ9CP12P5SKN5S5` | `32bc220a35ee` | 797 | `0effc543c750a331` |
| musique_clean/R1 | 2 | `fc-01M1ZY6HA3B6SJKSNKZXKFJM5H` | `32bc220a35ee` | 797 | `0effc543c750a331` |

Every seed of a cell scored one panel over one set of structural inputs: squad_clean/R1 has 1 panel digest and 1 shared-input digest across its three seeds; musique_clean/R1 has 1 panel digest and 1 shared-input digest across its three seeds. The holdout split carries no seed -- it is a fixed tail slice of a frozen-order split -- which is what makes a same-seed difference a difference in the seed and not in the queries.

Each new fit loaded its own seed's native S4 checkpoint from M2B's `resolution/seed{N}` subtree rather than the seed-0 headline tree, and re-scored it to M2B's filed recall@5 for that seed. A run that had silently read the headline checkpoint would have produced a well-formed artifact whose S4 column was constant across the three seeds being averaged; the runner refuses on that reproduction rather than reporting it.

## Training time and measured cost (sections 6 and 15b)

| blocker | seed | training s | peak train VRAM MB |
| --- | ---: | ---: | ---: |
| squad_clean/R1 | 1 | 159.86 | 1042.5 |
| squad_clean/R1 | 2 | 159.60 | 1043.9 |
| musique_clean/R1 | 1 | 24.80 | 365.7 |
| musique_clean/R1 | 2 | 25.03 | 365.0 |

Total measured training time across the four new fits: **369.29 s**.

Measured cost re-prices the compute record's own four line items with the seconds that happened, so prediction and measurement are comparable line by line. Unlike Stage 1's, this record predicted nothing: its seconds were read out of the Stage-1 artifacts for these same two cells and this same arm.

| blocker | seed | predicted s | measured s | ratio | re-score | fit | scoring | benchmark |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| squad_clean/R1 | 1 | 165.2 | 168.7 | 1.02 | 3.4 | 159.9 | 4.5 | 1.0 |
| squad_clean/R1 | 2 | 165.2 | 168.6 | 1.02 | 3.4 | 159.6 | 4.6 | 1.0 |
| musique_clean/R1 | 1 | 27.7 | 27.9 | 1.01 | 1.4 | 24.8 | 0.7 | 1.0 |
| musique_clean/R1 | 2 | 27.7 | 28.1 | 1.02 | 1.4 | 25.0 | 0.7 | 1.0 |
| **total** | | **385.8** | **393.3** | **1.02** | | | | |

At the record's own divisor (0.4) and rate ($2.2408/h): **$0.6121** compute plus $0.1896 measured container overhead = **$0.80 measured**, against $0.79 predicted and the $2.00 ceiling filed before launch. Within the ceiling: **true**. Within the stated 1.25x container safety factor: **true**.

## Failures and retries

Four jobs were spawned server-side through `scripts/spawn_modal_jobs.py` and four returned. No job failed, none was retried, none was resubmitted, and no abort criterion fired: no container exceeded its predicted walltime beyond the stated 1.25x factor, every container held the priced A10G, every panel matched the Stage-1 figure, and every native S4 re-score reproduced M2B's filed recall@5 for its own seed.

One physical run exists under each of the four prefixes, and the logical result was still selected by commit rather than by recency, because a fetch that takes the newest file reports an older run's numbers without noticing when a job does not land.

## Verdict

**STOP_S4_DEVELOPMENT_CONFIRMED**

musique_clean/R1 does not reach -0.50pp on the three-seed mean. The guard is section 12's and applies to both blockers.

**Then:** Freeze S3 as the development QLS model and proceed to M3 independent GNN development. No further S4 repair families.

What this closes is the S4 repair programme, not the measurement. S3 is retained as the development QLS model on the strength of the same rows that closed it, and the mechanism is now described rather than suspected: the 1,536-parameter semantic difference column recovers a large part of what the S4 widening cost -- most of it at rank 1 -- and still does not reach S3 on MuSiQue at any of three seeds.

## STOP_FOR_REVIEW

