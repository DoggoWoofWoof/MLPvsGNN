# ENC E2S: can mpnet's vectors be improved without touching mpnet? (report)

Written by `outputs/mp_unified/zenc2s.py report` under docs/ENC_ENCODER_COMPARISON.md section 14. R@k x100 on s1eval. Won back = the read minus E0's raw mpnet read, in points; gap = minus E0's frozen read; paired 95% bootstrap over questions (2,000, seed 0). Settings chosen on s1sel.

Chosen settings: {"win": {"W": 64, "beta": 0.5}, "fuse": {"w": 0.5}, "adapt": {"form": "linear", "lr": 0.0001, "epoch": 4}}

| arm | carve | R@5 | won back | gap to frozen | share of E0's gap won | R@20 gap | R@1000 gap | RRF@5 gap |
|---|---|---|---|---|---|---|---|---|
| win | squad | 78.56 | 5.05 [4.58, 5.53] | -9.13 [-9.78, -8.46] | 0.3561 | -5.57 | -0.76 | -4.58 |
| win | musique | 34.69 | 0.86 [0.38, 1.36] | -11.03 [-12.14, -9.98] | 0.0723 | -12.51 | -6.27 | -5.79 |
| fuse | squad | 76.48 | 2.96 [2.56, 3.39] | -11.22 [-11.88, -10.56] | 0.2087 | -6.63 | -0.72 | -5.66 |
| fuse | musique | 35.35 | 1.52 [0.96, 2.08] | -10.37 [-11.4, -9.34] | 0.1278 | -11.23 | -6.4 | -5.78 |
| adapt | squad | 72.64 | -0.88 [-1.38, -0.38] | -15.05 [-15.79, -14.33] | -0.0621 | -9.74 | -1.39 | -7.15 |
| adapt | musique | 38.29 | 4.46 [3.66, 5.29] | -7.43 [-8.44, -6.41] | 0.3751 | -9.12 | -7.12 | -3.67 |
| adapt (unseen) | hotpotqa | 46.01 | 3.38 [2.88, 3.89] | -19.76 [-20.5, -19.05] | 0.146 | -18.8 | -12.76 | -10.28 |

Calls (14.5): {"win": "IMPROVES", "fuse": "IMPROVES", "adapt": "NO"}

Beside, a pointer only (13.5's bars met by the gap to frozen): {"win": [], "fuse": [], "adapt": []}

Beside: cross maps (won back on the other dataset): {"cross-squad__musique": {"pts": 2.43, "ci": [1.88, 3.0], "call": "ABOVE"}, "cross-musique__squad": {"pts": -3.42, "ci": [-3.94, -2.89], "call": "BELOW"}}

Beside: the same map on the frozen vectors (against E0's frozen read): {"frozen-map__squad": {"pts": -0.87, "ci": [-1.24, -0.5], "call": "BELOW"}, "frozen-map__musique": {"pts": 2.17, "ci": [1.46, 2.91], "call": "ABOVE"}}

Three arms are read, so the chance that one passes by luck is above 5%.
