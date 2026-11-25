# SEM2 R6 report (docs/SEM2_EDGE_OPERATORS.md section 14)

Verdict: **ONE_WAY (passage -> KB)**. Missing: none.
Control, direction 4 Delta: ABOVE.

| direction | units | Delta name - cosine | Gain from the name | call | mean MRR cosine / mean name / name / shuffled |
|---|---|---|---|---|---|
| 1 KB -> KB (WebQSP net on MetaQA's S relations) | 7 | +0.79 [-0.21, +2.47] LEVEL | -0.13 [-1.51, +1.35] LEVEL | NO | 0.43 / 1.36 / 1.22 / 0.48 |
| 2 KB -> passage (WebQSP net on the 15 passage units) | 15 | -1.37 [-1.89, -0.85] BELOW | -0.80 [-1.43, -0.36] BELOW | LOSES | 8.90 / 8.34 / 7.54 / 7.88 |
| 3 passage -> KB (HotpotQA net on WebQSP's 163 and MetaQA's 7) | 169 | +1.53 [+0.68, +2.52] ABOVE | +0.15 [-0.08, +0.38] LEVEL | GENERIC | 2.52 / 3.90 / 4.05 / 4.03 |
| 4 passage -> passage, control (HotpotQA net on the other four corpora) | 11 | +2.36 [+1.06, +3.87] ABOVE | +0.15 [-0.39, +0.70] LEVEL | GENERIC | 9.28 / 11.49 / 11.64 / 10.45 |

## Beside (per dataset and per kind; deciding nothing)

| read | units | Delta | Gain | call | mean MRR c / m / n / s |
|---|---|---|---|---|---|
| kb on multihoprag (main units) | 1 | - | - | TOO_FEW_UNITS | 15.99 / 13.26 / 14.77 / 13.26 |
| kb on squad (main units) | 3 | -2.67 [-2.92, -2.51] BELOW | -1.90 [-4.28, -0.47] BELOW | LOSES | 15.91 / 15.14 / 13.24 / 14.08 |
| kb on musique (main units) | 3 | -1.03 [-1.60, -0.70] BELOW | -0.47 [-0.53, -0.39] BELOW | LOSES | 5.02 / 4.46 / 3.99 / 4.04 |
| kb on metaqa (main units) | 7 | +0.79 [-0.21, +2.47] LEVEL | -0.13 [-1.51, +1.35] LEVEL | NO | 0.43 / 1.36 / 1.22 / 0.48 |
| kb on metaqa (beside units: R labels) | 2 | - | - | TOO_FEW_UNITS | 12.27 / 9.61 / 10.38 / 9.90 |
| kb on hotpotqa (main units) | 4 | -1.27 [-2.13, -0.53] BELOW | -0.39 [-0.89, +0.16] LEVEL | LOSES | 7.86 / 6.98 / 6.59 / 6.83 |
| kb on 2wiki (main units) | 4 | -0.39 [-0.91, +0.26] LEVEL | -0.46 [-0.85, +0.03] LEVEL | NO | 5.83 / 5.90 / 5.44 / 5.83 |
| p on multihoprag (main units) | 1 | - | - | TOO_FEW_UNITS | 15.99 / 16.36 / 15.72 / 16.36 |
| p on squad (main units) | 3 | +3.09 [+1.10, +4.35] ABOVE | +0.61 [-0.87, +1.88] LEVEL | GENERIC | 15.91 / 18.39 / 19.00 / 16.94 |
| p on musique (main units) | 3 | +0.58 [+0.45, +0.67] ABOVE | -0.69 [-1.65, +0.09] LEVEL | GENERIC | 5.02 / 6.28 / 5.59 / 5.78 |
| p on metaqa (main units) | 7 | +1.45 [+0.16, +3.73] ABOVE | +0.08 [-0.08, +0.30] LEVEL | GENERIC | 0.43 / 1.81 / 1.89 / 1.70 |
| p on metaqa (beside units: R labels) | 2 | - | - | TOO_FEW_UNITS | 12.27 / 11.48 / 12.11 / 12.11 |
| p on webqsp (main units) | 162 | +1.53 [+0.65, +2.56] ABOVE | +0.15 [-0.07, +0.38] LEVEL | GENERIC | 2.61 / 3.99 / 4.14 / 4.13 |
| p on webqsp (beside units: R labels) | 3 | -0.37 [-3.64, +2.53] LEVEL | +1.11 [-1.23, +4.55] LEVEL | NO | 32.14 / 30.66 / 31.77 / 31.49 |
| p on hotpotqa (main units) | 4 | +4.12 [+1.50, +8.30] ABOVE | +1.04 [+0.32, +1.87] ABOVE | TRANSFERS | 7.86 / 10.94 / 11.98 / 9.43 |
| p on 2wiki (main units) | 4 | +3.64 [+1.11, +6.58] ABOVE | +0.30 [-0.04, +0.62] LEVEL | GENERIC | 5.83 / 9.17 / 9.47 / 7.62 |
| kb on R-label units (names seen) | 13 | -1.53 [-2.07, -0.96] BELOW | -0.82 [-1.49, -0.31] BELOW | LOSES | 9.19 / 8.48 / 7.66 / 8.00 |
| p on hotpotqa, in sample (X26) | 4 | +4.12 [+1.50, +8.30] ABOVE | +1.04 [+0.32, +1.87] ABOVE | TRANSFERS | 7.86 / 10.94 / 11.98 / 9.43 |

## S.hyperlink units under N_KB (the only passage-side name it has not seen; MRR points)

| unit | edges | cosine | name | mean name | shuffled |
|---|---|---|---|---|---|
| hotpotqa|S.hyperlink | 200 | 7.41 | 7.14 | 7.92 | 7.23 |
| 2wiki|S.hyperlink | 200 | 6.67 | 6.32 | 6.95 | 6.96 |

## Expectations (14.6; None = not yet readable)

- X22 KB -> KB Delta not ABOVE: True
- X23 KB -> passage Delta not ABOVE: True
- X23 hyperlink units: name no better than cosine on both: True
- X24 passage -> KB Delta not ABOVE: False
- X25 passage -> passage Delta ABOVE: True
- X26 N_P in sample Delta ABOVE: True
