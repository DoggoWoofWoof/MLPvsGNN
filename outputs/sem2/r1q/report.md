# SEM2 R1q: the query-bank normalization

Declared in docs/SEM2_EDGE_OPERATORS.md section 8.14. Written by zsem2hq.py report at 2026-11-24T17:30:10Z. MRR_T in points (each read group and target once); paired 95% bootstraps over those groups. Alphas chosen per read group on R1's dev sample.

| dataset | arm | rotation | rotation q | H5 (MRR_T) | H5 seen targets | H5 unseen targets | H5 on pooled | rotation alphas > 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multihoprag | e3v | 23.43 | 23.43 | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | none |
| multihoprag | frozen | 23.55 | 23.55 | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | none |
| squad | e3v | 27.9 | 27.9 | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | none |
| squad | frozen | 28.54 | 28.54 | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | none |
| musique | e3v | 12.1 | 12.1 | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | none |
| musique | frozen | 13.52 | 13.52 | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | none |
| metaqa | e3v | 10.23 | 10.21 | -0.02 [-0.03, +0.00] LEVEL | -0.02 [-0.04, -0.00] BELOW | +0.00 [-0.00, +0.01] LEVEL | -0.04 [-0.07, -0.01] BELOW | {'R.mention': 0.25} |
| metaqa | frozen | 10.89 | 10.84 | -0.05 [-0.10, -0.02] BELOW | -0.07 [-0.12, -0.02] BELOW | -0.00 [-0.01, +0.01] LEVEL | -0.05 [-0.09, -0.02] BELOW | {'S.written_by': 0.25} |
| webqsp | e3v | 49.3 | 49.3 | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | none |
| hotpotqa | e3v | 11.3 | 11.24 | -0.05 [-0.16, +0.05] LEVEL | -0.05 [-0.17, +0.06] LEVEL | -0.06 [-0.30, +0.21] LEVEL | -0.06 [-0.17, +0.05] LEVEL | {'R.mention': 0.25} |
| 2wiki | e3v | 9.55 | 9.55 | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | none |

**Carry (8.14, as 8.9: ABOVE on MetaQA, BELOW on none; e3v, MRR_T):** NOT_CARRIED (MetaQA LEVEL, BELOW on none)

**Expectations:** X10 False; X11 False
