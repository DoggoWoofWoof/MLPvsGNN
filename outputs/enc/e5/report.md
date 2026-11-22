# ENC E5: storage forms of e3v's document vectors (report)

Written by `outputs/mp_unified/zenc5.py report` under docs/ENC_ENCODER_COMPARISON.md section 21. Gaps are the form minus fp16 in points (s1eval), paired 95% bootstrap over questions.

**Call: int8** (1,152 bytes a document, 0.38 of frozen's).

| dataset | form | bytes | dense R@5 | R@5 gap | R@1000 gap | RRF R@5 gap | FREE |
|---|---|---|---|---|---|---|---|
| squad | fp16 | 2,304 | 83.69 | | | | |
| squad | int8 | 1,152 | 83.67 | -0.03 [-0.15, +0.08] LEVEL | -0.01 [-0.03, +0.00] LEVEL | +0.01 [-0.08, +0.10] LEVEL | yes |
| squad | pca768-own | 1,536 | 83.33 | -0.36 [-0.66, -0.05] BELOW | +0.00 [-0.09, +0.09] LEVEL | -0.27 [-0.51, -0.05] BELOW | no |
| squad | pca768-own+int8 | 768 | 83.34 | -0.35 [-0.65, -0.04] BELOW | +0.00 [-0.09, +0.09] LEVEL | -0.22 [-0.45, -0.01] BELOW | no |
| squad | pca768-loo | 1,536 | 83.19 | -0.51 [-0.80, -0.21] BELOW | -0.02 [-0.11, +0.07] LEVEL | -0.29 [-0.51, -0.07] BELOW | no |
| squad | pca768-loo+int8 | 768 | 83.21 | -0.48 [-0.77, -0.19] BELOW | -0.02 [-0.11, +0.07] LEVEL | -0.28 [-0.51, -0.06] BELOW | no |
| musique | fp16 | 2,304 | 47.69 | | | | |
| musique | int8 | 1,152 | 47.75 | +0.06 [-0.08, +0.19] LEVEL | +0.07 [-0.06, +0.21] LEVEL | +0.03 [-0.15, +0.21] LEVEL | yes |
| musique | pca768-own | 1,536 | 47.74 | +0.05 [-0.39, +0.48] LEVEL | +1.18 [+0.78, +1.56] ABOVE | -0.16 [-0.51, +0.20] LEVEL | yes |
| musique | pca768-own+int8 | 768 | 47.69 | +0.01 [-0.42, +0.43] LEVEL | +1.21 [+0.82, +1.61] ABOVE | -0.12 [-0.46, +0.22] LEVEL | yes |
| musique | pca768-loo | 1,536 | 47.64 | -0.05 [-0.49, +0.38] LEVEL | +0.98 [+0.56, +1.38] ABOVE | -0.39 [-0.74, -0.03] BELOW | no |
| musique | pca768-loo+int8 | 768 | 47.69 | +0.00 [-0.43, +0.43] LEVEL | +0.97 [+0.55, +1.37] ABOVE | -0.43 [-0.78, -0.08] BELOW | no |
| hotpotqa | fp16 | 2,304 | 65.35 | | | | |
| hotpotqa | int8 | 1,152 | 65.29 | -0.06 [-0.16, +0.04] LEVEL | -0.02 [-0.09, +0.05] LEVEL | -0.05 [-0.14, +0.04] LEVEL | yes |
| hotpotqa | pca768-own | 1,536 | 64.94 | -0.41 [-0.64, -0.17] BELOW | -0.09 [-0.30, +0.11] LEVEL | -0.54 [-0.74, -0.34] BELOW | no |
| hotpotqa | pca768-own+int8 | 768 | 64.91 | -0.44 [-0.67, -0.19] BELOW | -0.10 [-0.32, +0.11] LEVEL | -0.49 [-0.69, -0.28] BELOW | no |
| hotpotqa | pca768-loo | 1,536 | 64.85 | -0.49 [-0.72, -0.24] BELOW | -0.09 [-0.31, +0.11] LEVEL | -0.51 [-0.71, -0.30] BELOW | no |
| hotpotqa | pca768-loo+int8 | 768 | 64.87 | -0.48 [-0.71, -0.24] BELOW | -0.09 [-0.31, +0.11] LEVEL | -0.54 [-0.75, -0.33] BELOW | no |
| 2wiki | fp16 | 2,304 | 62.55 | | | | |
| 2wiki | int8 | 1,152 | 62.59 | +0.04 [-0.02, +0.10] LEVEL | -0.02 [-0.07, +0.03] LEVEL | +0.01 [-0.05, +0.06] LEVEL | yes |
| 2wiki | pca768-own | 1,536 | 62.04 | -0.51 [-0.67, -0.36] BELOW | +0.12 [-0.02, +0.24] LEVEL | -0.12 [-0.22, -0.02] BELOW | no |
| 2wiki | pca768-own+int8 | 768 | 62.03 | -0.52 [-0.68, -0.37] BELOW | +0.14 [+0.01, +0.26] ABOVE | -0.13 [-0.23, -0.02] BELOW | no |
| 2wiki | pca768-loo | 1,536 | 61.94 | -0.61 [-0.78, -0.45] BELOW | +0.10 [-0.03, +0.23] LEVEL | -0.16 [-0.27, -0.06] BELOW | no |
| 2wiki | pca768-loo+int8 | 768 | 61.93 | -0.62 [-0.79, -0.47] BELOW | +0.11 [-0.02, +0.24] LEVEL | -0.15 [-0.25, -0.04] BELOW | no |
| metaqa | fp16 | 2,304 | 0.65 | | | | |
| metaqa | int8 | 1,152 | 0.64 | -0.01 [-0.03, +0.00] LEVEL | -0.03 [-0.17, +0.10] LEVEL | +0.01 [-0.01, +0.03] LEVEL | yes |
| metaqa | pca768-own | 1,536 | 0.47 | -0.18 [-0.27, -0.09] BELOW | -4.74 [-5.18, -4.31] BELOW | -0.02 [-0.08, +0.03] LEVEL | no |
| metaqa | pca768-own+int8 | 768 | 0.48 | -0.17 [-0.26, -0.08] BELOW | -4.81 [-5.25, -4.38] BELOW | -0.01 [-0.06, +0.05] LEVEL | no |
| metaqa | pca768-loo | 1,536 | 0.47 | -0.18 [-0.28, -0.09] BELOW | -5.33 [-5.78, -4.89] BELOW | -0.03 [-0.08, +0.00] LEVEL | no |
| metaqa | pca768-loo+int8 | 768 | 0.47 | -0.18 [-0.28, -0.09] BELOW | -5.36 [-5.81, -4.91] BELOW | -0.03 [-0.08, +0.00] LEVEL | no |
| webqsp | fp16 | 2,304 | 6.73 | | | | |
| webqsp | int8 | 1,152 | 6.72 | -0.01 [-0.07, +0.03] LEVEL | -0.18 [-0.47, +0.10] LEVEL | +0.06 [-0.15, +0.32] LEVEL | yes |
| webqsp | pca768-own | 1,536 | 6.84 | +0.11 [-0.28, +0.52] LEVEL | +0.06 [-0.76, +0.92] LEVEL | +0.18 [-0.00, +0.42] LEVEL | yes |
| webqsp | pca768-own+int8 | 768 | 6.76 | +0.03 [-0.35, +0.39] LEVEL | +0.02 [-0.80, +0.88] LEVEL | +0.18 [-0.00, +0.42] LEVEL | yes |
| webqsp | pca768-loo | 1,536 | 6.44 | -0.29 [-0.74, +0.15] LEVEL | -3.72 [-4.77, -2.74] BELOW | +0.09 [-0.11, +0.34] LEVEL | no |
| webqsp | pca768-loo+int8 | 768 | 6.44 | -0.29 [-0.71, +0.11] LEVEL | -3.77 [-4.84, -2.78] BELOW | +0.09 [-0.11, +0.34] LEVEL | no |
