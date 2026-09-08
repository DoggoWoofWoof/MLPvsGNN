# M2D Stage 2 — gate

Applied by `scripts/m2d_stage2_gate.py`, committed before any Stage-2 fit existed. A3_MINIMAL at seed n against S3 at seed n, both blockers, seeds 0, 1 and 2. Seed 0 is Stage 1's own artifact and every S3 and S4 row is M2B's; nothing already fit was refit to produce a comparison row.

**Verdict: STOP_S4_DEVELOPMENT_CONFIRMED**

musique_clean/R1 does not reach -0.50pp on the three-seed mean. The guard is section 12's and applies to both blockers.

Authorised by: stage_2_amendment, filed 2026-09-08 by review.

Bound: -0.50pp on the arithmetic mean of the three same-seed differences, for **both** blockers. 4 of 4 new fits present; 6 of 6 same-seed rows.

## squad_clean/R1

| seed | A3 R@1 | A3 R@5 | A3 R@20 | A3 MRR | A3-S3 R@5 pp | A3-S4 R@5 pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 * | 0.726453 | 0.918473 | 0.967197 | 0.810455 | +0.077 | +0.825 |
| 1 | 0.726837 | 0.915787 | 0.966046 | 0.810126 | -0.096 | +0.902 |
| 2 | 0.728180 | 0.915979 | 0.965663 | 0.811147 | -0.307 | +0.595 |

Three-seed mean A3-S3 recall@5: **-0.109pp** (sample SD 0.192pp, signs +--). Meets -0.50pp: **yes**.

| metric | mean A3-S3 pp | mean A3-S4 pp |
| --- | ---: | ---: |
| recall@1 | -0.237 | +4.067 |
| recall@5 | -0.109 | +0.774 |
| recall@20 | -0.051 | +0.211 |
| mrr | -0.151 | +2.464 |

Identity holds: true (seeds [1, 2]).

## musique_clean/R1

| seed | A3 R@1 | A3 R@5 | A3 R@20 | A3 MRR | A3-S3 R@5 pp | A3-S4 R@5 pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 * | 0.397114 | 0.767880 | 0.871393 | 0.873413 | -0.565 | +1.694 |
| 1 | 0.398369 | 0.760979 | 0.868256 | 0.874406 | -1.192 | +1.255 |
| 2 | 0.415935 | 0.757842 | 0.865119 | 0.894080 | -1.506 | +1.506 |

Three-seed mean A3-S3 recall@5: **-1.087pp** (sample SD 0.479pp, signs ---). Meets -0.50pp: **no**.

| metric | mean A3-S3 pp | mean A3-S4 pp |
| --- | ---: | ---: |
| recall@1 | -3.994 | +8.511 |
| recall@5 | -1.087 | +1.485 |
| recall@20 | -0.774 | +0.335 |
| mrr | -4.715 | +10.207 |

Identity holds: true (seeds [1, 2]).

## What this gate does not say

The per-seed signs and the sample SD above are reported and do not gate. No latency result is claimed: Stage 2 ran no benchmark, and Stage 1's p95 figures remain Stage 1's.

**Then:** Freeze S3 as the development QLS model and proceed to M3 independent GNN development. No further S4 repair families.

**STOP_FOR_REVIEW**
