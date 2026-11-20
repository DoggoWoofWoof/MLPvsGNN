# ENC E3W: the models on e3v's vectors (gte-modernbert + granite-small, fused), in-domain (report)

Written by `outputs/mp_unified/zenc2r.py report` under docs/ENC_ENCODER_COMPARISON.md section 19 (sections 12.9 to 12.11's rule). R@5 x100 on s1eval, candidate p@swa; a gap is the arm's world minus the frozen world in points, paired 95% bootstrap over questions. GNX and zrckpn decide; zg6r is GNX's stage, read beside. The user decides (D101).

GA: frozen PASS (22 runs), e3v PASS (50 runs)

## e3v

| model | carve | frozen | arm | model gap | first-stage gap | won back | in-pool RRF gap |
|---|---|---|---|---|---|---|---|
| zg6r (GNN stage, beside) | squad | 91.13 | 89.66 | -1.47 [-1.84, -1.08] BELOW | -4.00 [-4.56, -3.46] BELOW | 2.53 | -2.29 [-2.66, -1.93] BELOW |
| zg6r (GNN stage, beside) | musique | 57.53 | 58.2 | +0.68 [-0.21, +1.52] LEVEL | +1.97 [+1.01, +2.88] ABOVE | -1.29 | +0.43 [-0.19, +1.03] LEVEL |
| GNX (GNN) | squad | 91.05 | 89.71 | -1.34 [-1.71, -0.94] BELOW | -4.00 [-4.56, -3.46] BELOW | 2.66 | -2.29 [-2.66, -1.93] BELOW |
| GNX (GNN) | musique | 58.21 | 57.55 | -0.66 [-1.50, +0.12] LEVEL | +1.97 [+1.01, +2.88] ABOVE | -2.63 | +0.43 [-0.19, +1.03] LEVEL |
| zrckpn (MLP) | squad | 91.42 | 89.67 | -1.75 [-2.12, -1.39] BELOW | -4.00 [-4.56, -3.46] BELOW | 2.25 | -2.29 [-2.66, -1.93] BELOW |
| zrckpn (MLP) | musique | 57.95 | 57.48 | -0.47 [-1.25, +0.32] LEVEL | +1.97 [+1.01, +2.88] ABOVE | -2.44 | +0.43 [-0.19, +1.03] LEVEL |

- squad pools: golds held 97.98 (frozen) vs 97.4 (e3v), -0.58 [-0.76, -0.41] BELOW; mean size 50.0 vs 50.0
- musique pools: golds held 97.77 (frozen) vs 97.35 (e3v), -0.42 [-0.71, -0.14] BELOW; mean size 5263.6 vs 5320.6

Calls (11.5): gnx LOSES, zrckpn LOSES

## Cost

- frozen: {"params": null, "dim": 1536, "bytes_per_row_f16": 3072, "padded_to": 1536, "audited_seconds": {"other": 391.4, "fits": 13061.8, "reads": 1052.8}}
- e3v: {"params": 196676736, "dim": 1152, "bytes_per_row_f16": 2304, "padded_to": 1536, "audited_seconds": {"other": 24349.0, "fits": 11274.2, "reads": 848.1}, "params_parts": {"gtemodern": 149014272, "granitesm": 47662464}}

## Beside: E2's arms (outputs/enc/e2/report.json)

| arm | model | carve | frozen | arm | model gap | won back |
|---|---|---|---|---|---|---|
| mpnet | GNX (GNN) | musique | 58.21 | 53.07 | -5.14 [-6.06, -4.23] BELOW | 6.75 |
| mpnet | GNX (GNN) | squad | 91.05 | 88.43 | -2.62 [-3.02, -2.16] BELOW | 11.56 |
| mpnet | zg6r (GNN stage, beside) | musique | 57.53 | 53.28 | -4.25 [-5.18, -3.35] BELOW | 7.64 |
| mpnet | zg6r (GNN stage, beside) | squad | 91.13 | 88.54 | -2.59 [-3.01, -2.15] BELOW | 11.59 |
| mpnet | zrckpn (MLP) | musique | 57.95 | 54.13 | -3.82 [-4.72, -2.95] BELOW | 8.07 |
| mpnet | zrckpn (MLP) | squad | 91.42 | 88.65 | -2.77 [-3.18, -2.35] BELOW | 11.41 |
| qwen3e06 | GNX (GNN) | musique | 58.21 | 54.63 | -3.58 [-4.47, -2.71] BELOW | 0.88 |
| qwen3e06 | GNX (GNN) | squad | 91.05 | 89.56 | -1.49 [-1.88, -1.09] BELOW | 4.84 |
| qwen3e06 | zg6r (GNN stage, beside) | musique | 57.53 | 55.33 | -2.19 [-3.10, -1.35] BELOW | 2.27 |
| qwen3e06 | zg6r (GNN stage, beside) | squad | 91.13 | 89.62 | -1.51 [-1.90, -1.11] BELOW | 4.82 |
| qwen3e06 | zrckpn (MLP) | musique | 57.95 | 54.03 | -3.91 [-4.76, -3.10] BELOW | 0.55 |
| qwen3e06 | zrckpn (MLP) | squad | 91.42 | 89.75 | -1.67 [-2.03, -1.29] BELOW | 4.66 |

E2's calls: mpnet: gnx LOSES, zrckpn LOSES; qwen3e06: gnx LOSES, zrckpn LOSES
