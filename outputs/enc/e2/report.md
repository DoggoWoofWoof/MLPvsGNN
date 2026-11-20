# ENC E2: the models on the arm's vectors, in-domain (report)

Written by `outputs/mp_unified/zenc2r.py report` under docs/ENC_ENCODER_COMPARISON.md sections 12.9, 12.10 and 12.11. R@5 x100 on s1eval, candidate p@swa; a gap is the arm's world minus the frozen world in points, paired 95% bootstrap over questions. GNX and zrckpn decide; zg6r is GNX's stage, read beside. The user decides (D101).

GA: frozen PASS (22 runs), mpnet PASS (50 runs), qwen3e06 PASS (50 runs)

## mpnet

| model | carve | frozen | arm | model gap | first-stage gap | won back | in-pool RRF gap |
|---|---|---|---|---|---|---|---|
| zg6r (GNN stage, beside) | squad | 91.13 | 88.54 | -2.59 [-3.01, -2.15] BELOW | -14.18 [-14.87, -13.47] BELOW | 11.59 | -6.55 [-7.04, -6.06] BELOW |
| zg6r (GNN stage, beside) | musique | 57.53 | 53.28 | -4.25 [-5.18, -3.35] BELOW | -11.89 [-12.99, -10.81] BELOW | 7.64 | -6.12 [-6.92, -5.30] BELOW |
| GNX (GNN) | squad | 91.05 | 88.43 | -2.62 [-3.02, -2.16] BELOW | -14.18 [-14.87, -13.47] BELOW | 11.56 | -6.55 [-7.04, -6.06] BELOW |
| GNX (GNN) | musique | 58.21 | 53.07 | -5.14 [-6.06, -4.23] BELOW | -11.89 [-12.99, -10.81] BELOW | 6.75 | -6.12 [-6.92, -5.30] BELOW |
| zrckpn (MLP) | squad | 91.42 | 88.65 | -2.77 [-3.18, -2.35] BELOW | -14.18 [-14.87, -13.47] BELOW | 11.41 | -6.55 [-7.04, -6.06] BELOW |
| zrckpn (MLP) | musique | 57.95 | 54.13 | -3.82 [-4.72, -2.95] BELOW | -11.89 [-12.99, -10.81] BELOW | 8.07 | -6.12 [-6.92, -5.30] BELOW |

- squad pools: golds held 97.98 (frozen) vs 96.88 (mpnet), -1.09 [-1.31, -0.88] BELOW; mean size 50.0 vs 50.0
- musique pools: golds held 97.77 (frozen) vs 97.22 (mpnet), -0.55 [-0.85, -0.24] BELOW; mean size 5263.6 vs 5254.0

Calls (11.5): gnx LOSES, zrckpn LOSES

## qwen3e06

| model | carve | frozen | arm | model gap | first-stage gap | won back | in-pool RRF gap |
|---|---|---|---|---|---|---|---|
| zg6r (GNN stage, beside) | squad | 91.13 | 89.62 | -1.51 [-1.90, -1.11] BELOW | -6.33 [-6.91, -5.75] BELOW | 4.82 | -3.39 [-3.80, -3.00] BELOW |
| zg6r (GNN stage, beside) | musique | 57.53 | 55.33 | -2.19 [-3.10, -1.35] BELOW | -4.46 [-5.41, -3.52] BELOW | 2.27 | -2.33 [-3.00, -1.61] BELOW |
| GNX (GNN) | squad | 91.05 | 89.56 | -1.49 [-1.88, -1.09] BELOW | -6.33 [-6.91, -5.75] BELOW | 4.84 | -3.39 [-3.80, -3.00] BELOW |
| GNX (GNN) | musique | 58.21 | 54.63 | -3.58 [-4.47, -2.71] BELOW | -4.46 [-5.41, -3.52] BELOW | 0.88 | -2.33 [-3.00, -1.61] BELOW |
| zrckpn (MLP) | squad | 91.42 | 89.75 | -1.67 [-2.03, -1.29] BELOW | -6.33 [-6.91, -5.75] BELOW | 4.66 | -3.39 [-3.80, -3.00] BELOW |
| zrckpn (MLP) | musique | 57.95 | 54.03 | -3.91 [-4.76, -3.10] BELOW | -4.46 [-5.41, -3.52] BELOW | 0.55 | -2.33 [-3.00, -1.61] BELOW |

- squad pools: golds held 97.98 (frozen) vs 97.09 (qwen3e06), -0.88 [-1.08, -0.70] BELOW; mean size 50.0 vs 50.0
- musique pools: golds held 97.77 (frozen) vs 97.19 (qwen3e06), -0.58 [-0.86, -0.30] BELOW; mean size 5263.6 vs 5415.9

Calls (11.5): gnx LOSES, zrckpn LOSES

## Cost

- frozen: {"params": null, "dim": 1536, "bytes_per_row_f16": 3072, "padded_to": 1536, "audited_seconds": {"other": 391.4, "fits": 13061.8, "reads": 1052.8}}
- mpnet: {"params": 109486464, "dim": 768, "bytes_per_row_f16": 1536, "padded_to": 1536, "audited_seconds": {"other": 24131.9, "fits": 13162.6, "reads": 1121.2}}
- qwen3e06: {"params": 595776512, "dim": 1024, "bytes_per_row_f16": 2048, "padded_to": 1536, "audited_seconds": {"other": 26900.2, "fits": 11691.3, "reads": 1311.8}}
