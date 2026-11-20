# ENC E0N report

Declared in docs/ENC_ENCODER_COMPARISON.md section 17. Dense R@5 on E0's s1eval carves; points; paired 95% bootstrap over questions (2,000, seed 0).

| arm | call | SQuAD R@5 | MuSiQue R@5 | vs mpnet (SQuAD / MuSiQue) | vs mpnet V | vs qwen3e06 | vs frozen |
|---|---|---|---|---|---|---|---|
| gtemodern | **STRONG** | 83.09 | 44.27 | +9.57 [+8.89, +10.28] / +10.44 [+9.39, +11.48] | +7.66 [+7.01, +8.32] / +4.96 [+3.97, +6.00] | +1.72 [+1.12, +2.28] / +3.01 [+2.12, +3.88] | -4.61 [-5.19, -4.02] / -1.45 [-2.38, -0.57] |
| granite | **STRONG** | 77.38 | 44.63 | +3.86 [+3.18, +4.53] / +10.80 [+9.78, +11.80] | +1.95 [+1.30, +2.59] / +5.32 [+4.35, +6.26] | -3.99 [-4.62, -3.38] / +3.37 [+2.46, +4.27] | -10.32 [-10.97, -9.65] / -1.09 [-2.12, -0.13] |
| granitesm | **STRONG** | 76.12 | 43.89 | +2.60 [+1.90, +3.34] / +10.06 [+9.02, +11.07] | +0.69 [+0.02, +1.36] / +4.59 [+3.66, +5.52] | -5.25 [-5.90, -4.62] / +2.63 [+1.79, +3.44] | -11.57 [-12.26, -10.90] / -1.82 [-2.81, -0.87] |

References: frozen 87.69 / 45.72; mpnet 73.52 / 33.83; qwen3e06 81.37 / 41.26; mpnet-V 75.43 / 39.31

## Beside (no call)

| arm | ds | R@20 | R@1000 | RRF@5 | gold gap at 5 / 1000 | order share |
|---|---|---|---|---|---|---|
| gtemodern | squad | 91.43 | 99.28 | 88.39 | 547 / 57 | 0.896 |
| gtemodern | musique | 55.97 | 82.69 | 47.66 | 63 / 308 | -3.889 |
| granite | squad | 87.90 | 98.97 | 85.16 | 1225 / 93 | 0.924 |
| granite | musique | 58.03 | 85.49 | 46.49 | 52 / 97 | -0.865 |
| granitesm | squad | 87.25 | 98.91 | 84.22 | 1374 / 101 | 0.926 |
| granitesm | musique | 56.15 | 84.27 | 46.21 | 98 / 166 | -0.694 |
| frozen (ref) | both | 94.70 / 59.91 | 99.76 / 87.25 | 90.53 / 47.34 | | |
| mpnet (ref) | both | 86.45 / 47.07 | 98.73 / 79.97 | 83.98 / 41.24 | | |
| qwen3e06 (ref) | both | 90.52 / 53.98 | 99.15 / 83.60 | 87.17 / 44.98 | | |
| mpnet-V (ref) | both | 86.97 / 51.66 | 98.85 / 81.08 | 84.57 / 43.92 | | |

MuSiQue R@5 by hop count: gtemodern {2: 52.88, 3: 39.87, 4: 25.93}; granite {2: 53.35, 3: 41.4, 4: 23.7}; granitesm {2: 52.8, 3: 39.96, 4: 23.77}; frozen {2: 55.47, 3: 41.45, 4: 23.58}; mpnet {2: 43.97, 3: 27.98, 4: 13.46}; qwen3e06 {2: 50.2, 3: 37.76, 4: 20.19}; mpnet-V {2: 47.48, 3: 36.62, 4: 19.07}

Bars (pointers only): {"mpnet": {"zrckpn": {"squad": -5.12, "musique": -3.11}, "gnx": {"squad": -5.41, "musique": -2.31}}, "qwen3e06": {"zrckpn": {"squad": -3.79, "musique": -1.14}, "gnx": {"squad": -4.25, "musique": -1.25}}}

Clears the bars (gap to frozen, both carves): {"gtemodern": {"mpnet": {"zrckpn": true, "gnx": true}, "qwen3e06": {"zrckpn": false, "gnx": false}}, "granite": {"mpnet": {"zrckpn": false, "gnx": false}, "qwen3e06": {"zrckpn": false, "gnx": false}}, "granitesm": {"mpnet": {"zrckpn": false, "gnx": false}, "qwen3e06": {"zrckpn": false, "gnx": false}}}

Cost: {"gtemodern": {"params": 149014272, "dim": 768, "encode_seconds": 319.1, "cut": 0}, "granite": {"params": 149014272, "dim": 768, "encode_seconds": 320.2, "cut": 0}, "granitesm": {"params": 47662464, "dim": 384, "encode_seconds": 67.0, "cut": 0}}

Next, best first (17.6): gtemodern, granite, granitesm
