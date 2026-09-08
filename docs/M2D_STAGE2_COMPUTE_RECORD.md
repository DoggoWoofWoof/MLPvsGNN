# M2D Stage 2 — compute record

Filed at `43392d193e39`, before any Stage-2 job was submitted. 4 GPU jobs, one per cell and seed, each fitting A3_MINIMAL once and re-scoring M2B's native S4 on the same panel -- 4 new fits in total.

Authorised by: stage_2_amendment, filed 2026-09-08 by review.

## Why this is not Stage 1's number

Stage 1's record predicted: it scaled M2B's measured S4 seconds by a forward multiplier measured on the host, because no M2D arm had ever been fit. That prediction has since been tested. Every second below is read from the Stage-1 artifacts for these two cells and this arm, so the basis is an observation rather than a model of one -- and the workload is half the size per container, because A1 is not rerun.

## Container

| | |
| --- | --- |
| GPU | A10G |
| Cores | 16 |
| Memory | 49152 MB |
| Timeout | 3600 s |
| Rate | $2.2408/h |

Stage 2 trains. Four fits of a 206,785-parameter scorer over panels of up to 20,850 training queries is a backward pass per step, and it is the shape the seconds below were measured on.

## Workload

| cell | seed | fit s | scoring s | native re-score s | benchmark s | total s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| squad_clean/R1 | 1 | 156.43 | 4.43 | 3.32 | 1.03 | 165.21 |
| squad_clean/R1 | 2 | 156.43 | 4.43 | 3.32 | 1.03 | 165.21 |
| musique_clean/R1 | 1 | 24.48 | 0.69 | 1.45 | 1.05 | 27.67 |
| musique_clean/R1 | 2 | 24.48 | 0.69 | 1.45 | 1.05 | 27.67 |

Seed 0 is Stage 1's artifact and is not refit. S3 and S4 are M2B's at all three seeds and are not refit. Native S4 is loaded from M2B's checkpoint and re-scored, which is a forward pass and is priced as one.

No forward multiplier, no scaling of another rung's seconds, no host-side kernel timing. The four line items per container are the four Stage 1 measured.

## Prediction

| | |
| --- | --- |
| New fits | 4 |
| Jobs | 4 |
| GPU seconds | 385.8 |
| Largest job | squad_clean/R1 seed 1 at 165.2 s |
| Walltime within timeout | true |
| Safety factor | 1.25 |
| Utilisation assumed | 0.4 |
| Compute | $0.6003 |
| Container overhead | $0.1896 |
| **Expected spend** | **$0.79** |
| At the safety factor | $0.94 |
| **Hard ceiling** | **$2.00** |
| Peak train VRAM measured | 1042 MB |

Section 6 forbids a large multiplier without justification. Stage 1's 2x covered a predicted workload; this one is measured, on the same code, panel and box, so the only residual is seed-to-seed training variation and container variance.

Storage: Four artifacts and four checkpoints on the existing result volume, under seed-bearing paths. No new volume, no new dataset copy, and nothing that could overwrite a Stage-1 result.

Container overhead source: outputs/m2_qls_v2_freeze/measured_cost.json, measured over 4 containers

## Abort criteria

- any job exceeding its predicted walltime above by more than the stated safety factor of 1.25
- a container acquiring an accelerator other than the one priced here, which has no measured rate in this project
- a cell whose panel size differs from the Stage-1 figure above, which would mean the same-seed comparison is not against the same panel
- a native S4 re-score that does not reproduce M2B's filed recall@5 within the cell's stated bound, which the runner refuses on
- cumulative spend reaching the hard ceiling

## What this record does not authorise

- a fourth seed
- A1, or any arm other than A3_MINIMAL
- any cell other than squad_clean/R1 and musique_clean/R1
- the full 14-cell M2D screen
- any refit of seed 0, S3 or S4
- reading the test split
