# ENC E3B report

Declared in docs/ENC_ENCODER_COMPARISON.md section 18. s1eval dense R@5, points, paired 95% bootstrap over questions (2,000, seed 0); SQuAD / MuSiQue.

| arm | params / width | recipe | setting | R@5 | won back (vs raw) | call | gap to frozen | gate (V) |
|---|---|---|---|---|---|---|---|---|
| gtemodern | 149M / 768 | V | `{"gamma": 0.5, "beta": 0.0, "W": null, "w": 0.0, "prf": null}` | 82.93 / 45.18 | -0.16 [-0.44, +0.11] / +0.91 [+0.48, +1.36] | MIXED | -4.77 [-5.35, -4.21] / -0.54 [-1.49, +0.35] | NEAR |
| gtemodern | 149M / 768 | F | `{"gamma": 0.5, "beta": 0.75, "W": 64, "w": 0.0, "prf": null}` | 88.15 / 45.95 | +5.06 [+4.59, +5.54] / +1.68 [+1.03, +2.30] | IMPROVES | +0.45 [-0.08, +0.99] / +0.23 [-0.69, +1.13] |  |
| gtemodern+granite | 298M / 1536 | V | `{"gamma": 0.5, "beta": 0.0, "W": null, "w": 0.5, "prf": null}` | 83.53 / 49.00 | +0.44 [+0.00, +0.88] / +4.73 [+4.05, +5.40] | MIXED | -4.17 [-4.72, -3.63] / +3.28 [+2.38, +4.18] | ADVANCE |
| gtemodern+granite | 298M / 1536 | F | `{"gamma": 0.5, "beta": 0.75, "W": 64, "w": 0.5, "prf": null}` | 88.71 / 49.09 | +5.62 [+5.12, +6.12] / +4.82 [+4.10, +5.57] | IMPROVES | +1.01 [+0.49, +1.54] / +3.37 [+2.43, +4.29] |  |
| gtemodern+granitesm | 197M / 1152 | V | `{"gamma": 0.5, "beta": 0.0, "W": null, "w": 0.5, "prf": null}` | 83.68 / 48.15 | +0.59 [+0.19, +0.99] / +3.88 [+3.23, +4.47] | IMPROVES | -4.02 [-4.57, -3.49] / +2.43 [+1.53, +3.29] | ADVANCE |
| gtemodern+granitesm | 197M / 1152 | F | `{"gamma": 0.5, "beta": 0.75, "W": 64, "w": 0.5, "prf": null}` | 88.93 / 48.62 | +5.85 [+5.35, +6.33] / +4.35 [+3.63, +5.07] | IMPROVES | +1.24 [+0.70, +1.77] / +2.91 [+1.96, +3.83] |  |
| granitesm | 48M / 384 | V | `{"gamma": 0.0, "beta": 0.0, "W": null, "w": 0.0, "prf": null}` | 76.12 / 43.89 | +0.00 [+0.00, +0.00] / +0.00 [+0.00, +0.00] | NO | -11.57 [-12.26, -10.90] / -1.82 [-2.81, -0.87] | SHORT |
| granitesm | 48M / 384 | F | `{"gamma": 0.0, "beta": 0.5, "W": 64, "w": 0.0, "prf": null}` | 82.59 / 44.62 | +6.47 [+5.95, +6.97] / +0.72 [+0.24, +1.20] | IMPROVES | -5.10 [-5.69, -4.49] / -1.10 [-2.10, -0.13] |  |

## Beside (no call)

- gtemodern V: R@20 91.55 / 57.06, R@1000 99.42 / 83.87, RRF@5 88.41 / 47.97; leave one out gamma -0.16 [-0.44, +0.11] / +0.91 [+0.48, +1.36]; against qwen3e06 V +0.92 [+0.32, +1.46] / +2.68 [+1.81, +3.54], frozen V -4.82 [-5.40, -4.27] / -2.45 [-3.35, -1.59]; clears {"mpnet": "zrckpn", "qwen3e06": null}
- gtemodern F: R@20 94.66 / 58.25, R@1000 99.72 / 84.21, RRF@5 90.80 / 48.18; leave one out gamma +0.14 [-0.08, +0.38] / +1.32 [+0.89, +1.75], beta +5.22 [+4.74, +5.69] / +0.77 [+0.19, +1.36]; against qwen3e06 F +1.07 [+0.55, +1.54] / +2.18 [+1.33, +3.04], frozen F -2.30 [-2.75, -1.83] / -2.62 [-3.50, -1.67]; clears {"mpnet": "zrckpn", "qwen3e06": "zrckpn"}
- gtemodern+granite V: R@20 91.89 / 60.51, R@1000 99.50 / 86.81, RRF@5 88.07 / 48.02; leave one out gamma -0.30 [-0.54, -0.06] / +1.02 [+0.56, +1.47], w +0.60 [+0.19, +1.03] / +3.82 [+3.19, +4.47]; against qwen3e06 V +1.52 [+0.99, +2.04] / +6.50 [+5.66, +7.37], frozen V -4.22 [-4.78, -3.67] / +1.37 [+0.52, +2.22]; clears {"mpnet": "zrckpn", "qwen3e06": "gnx"}
- gtemodern+granite F: R@20 95.01 / 61.92, R@1000 99.78 / 86.94, RRF@5 90.56 / 48.47; leave one out gamma -0.06 [-0.29, +0.17] / +0.76 [+0.30, +1.18], beta +5.18 [+4.70, +5.64] / +0.09 [-0.43, +0.61], w +0.56 [+0.23, +0.88] / +3.14 [+2.52, +3.76]; against qwen3e06 F +1.63 [+1.14, +2.09] / +5.32 [+4.48, +6.14], frozen F -1.74 [-2.21, -1.30] / +0.52 [-0.41, +1.39]; clears {"mpnet": "zrckpn", "qwen3e06": "zrckpn"}
- gtemodern+granitesm V: R@20 92.07 / 59.80, R@1000 99.50 / 86.29, RRF@5 88.29 / 48.01; leave one out gamma -0.03 [-0.24, +0.19] / +0.43 [+0.08, +0.79], w +0.75 [+0.37, +1.13] / +2.97 [+2.38, +3.51]; against qwen3e06 V +1.67 [+1.12, +2.19] / +5.65 [+4.79, +6.49], frozen V -4.07 [-4.63, -3.53] / +0.52 [-0.37, +1.35]; clears {"mpnet": "zrckpn", "qwen3e06": "gnx"}
- gtemodern+granitesm F: R@20 95.12 / 61.15, R@1000 99.76 / 86.58, RRF@5 90.60 / 48.41; leave one out gamma -0.11 [-0.29, +0.06] / +0.49 [+0.12, +0.87], beta +5.26 [+4.78, +5.74] / +0.48 [-0.07, +1.06], w +0.78 [+0.49, +1.08] / +2.68 [+2.05, +3.28]; against qwen3e06 F +1.85 [+1.36, +2.32] / +4.86 [+4.02, +5.68], frozen F -1.52 [-1.96, -1.07] / +0.06 [-0.84, +0.94]; clears {"mpnet": "zrckpn", "qwen3e06": "zrckpn"}
- granitesm V: R@20 87.25 / 56.15, R@1000 98.91 / 84.27, RRF@5 84.22 / 46.21; leave one out none; against qwen3e06 V -5.89 [-6.54, -5.26] / +1.40 [+0.57, +2.23], frozen V -11.62 [-12.31, -10.93] / -3.73 [-4.67, -2.85]; clears {"mpnet": null, "qwen3e06": null}
- granitesm F: R@20 91.63 / 57.52, R@1000 99.46 / 85.12, RRF@5 87.75 / 46.60; leave one out beta +6.47 [+5.95, +6.97] / +0.72 [+0.24, +1.20]; against qwen3e06 F -4.49 [-5.06, -3.90] / +0.86 [-0.05, +1.70], frozen F -7.86 [-8.45, -7.24] / -3.95 [-4.93, -3.02]; clears {"mpnet": "zrckpn", "qwen3e06": null}

Bars (pointers): {"mpnet": {"zrckpn": {"squad": -5.12, "musique": -3.11}, "gnx": {"squad": -5.41, "musique": -2.31}}, "qwen3e06": {"zrckpn": {"squad": -3.79, "musique": -1.14}, "gnx": {"squad": -4.25, "musique": -1.25}}}

Next (18.7): {"call": "ADVANCE", "best": "gtemodern+granite", "smaller_within_0.5": "gtemodern+granitesm"}
