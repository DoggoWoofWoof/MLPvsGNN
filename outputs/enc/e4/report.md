# ENC E4: e3v's first stage on the six datasets, beside frozen (report)

Written by `outputs/mp_unified/zenc4.py report` under docs/ENC_ENCODER_COMPARISON.md section 20. s1eval dense R@k x100; a gap is e3v minus frozen in points, paired 95% bootstrap over questions. Beside, no call.

| dataset | questions | frozen R@5 | e3v R@5 | R@5 gap | R@20 gap | R@1000 gap |
|---|---|---|---|---|---|---|
| squad | 11,873 | 87.69 | 83.69 | -4.00 [-4.56, -3.46] BELOW | -2.65 [-3.06, -2.26] BELOW | -0.30 [-0.44, -0.18] BELOW |
| musique | 2,417 | 45.72 | 47.69 | +1.97 [+1.01, +2.88] ABOVE | -0.40 [-1.28, +0.43] LEVEL | -1.89 [-2.55, -1.24] BELOW |
| hotpotqa | 7,405 | 65.77 | 65.35 | -0.43 [-0.96, +0.13] LEVEL | -1.26 [-1.75, -0.76] BELOW | -0.68 [-1.06, -0.30] BELOW |
| 2wiki | 12,576 | 51.71 | 62.55 | +10.85 [+10.41, +11.29] ABOVE | +6.82 [+6.46, +7.23] ABOVE | +0.20 [-0.07, +0.48] LEVEL |
| webqsp | 1,503 | 10.95 | 6.73 | -4.22 [-5.32, -3.17] BELOW | -6.22 [-7.63, -4.91] BELOW | -14.12 [-16.08, -12.18] BELOW |
| metaqa | 9,785 | 0.87 | 0.65 | -0.22 [-0.38, -0.04] BELOW | -0.49 [-0.75, -0.23] BELOW | +4.94 [+4.15, +5.75] ABOVE |

FC: hotpotqa PASS, 2wiki PASS, webqsp PASS, metaqa PASS
