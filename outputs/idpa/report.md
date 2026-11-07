# IDPA: GNX (the same inputs as the MLP of record) on HippoRAG 2's settings, beside zrckpn and zg6r

Declared in docs/IDP_IDENTICAL_INPUT_PAIR.md section 5. R@5 in points, the 3-seed mean over the splits of the domain (95% interval of the questions, 2000 resamples, seed 0). Differences are GNX minus the peer, with the interval of the paired per-question difference, a label by it, and the fit pairs where GNX is above. No threshold, no adoption call.

| setting | domain | carve | zrckpn | zg6r | GNX | GNX - zrckpn | label | pairs above | GNX - zg6r | label | pairs above |
|---|---|---|---:|---:|---:|---|---|---|---|---|---|
| musique | in-domain | b1w | 67.4 | 69.0 | 69.1 | +1.6 (+0.9, +2.4) | GNX_ABOVE | 3/3 | +0.1 (-0.5, +0.7) | LEVEL | 2/3 |
| musique | in-domain | b1t | 64.5 | 66.0 | 66.2 | +1.8 (+1.0, +2.5) | GNX_ABOVE | 3/3 | +0.3 (-0.3, +0.8) | LEVEL | 2/3 |
| musique | zero-shot | b1w | 61.8 | 60.8 | 63.1 | +1.2 (+0.4, +2.1) | GNX_ABOVE | 3/3 | +2.3 (+1.6, +2.9) | GNX_ABOVE | 3/3 |
| musique | zero-shot | b1t | 60.0 | 62.1 | 63.0 | +3.0 (+2.1, +3.8) | GNX_ABOVE | 3/3 | +0.9 (+0.2, +1.7) | GNX_ABOVE | 2/3 |
| 2wiki | in-domain | b1w | 94.1 | 94.9 | 95.0 | +0.9 (+0.6, +1.2) | GNX_ABOVE | 5/6 | +0.1 (-0.1, +0.3) | LEVEL | 5/6 |
| 2wiki | in-domain | b1t | 77.0 | 78.1 | 77.6 | +0.6 (+0.2, +1.0) | GNX_ABOVE | 5/6 | -0.5 (-0.8, -0.2) | OTHER_ABOVE | 0/6 |
| hotpotqa | in-domain | b1w | 93.4 | 94.3 | 94.1 | +0.7 (+0.2, +1.1) | GNX_ABOVE | 3/3 | -0.2 (-0.5, +0.1) | LEVEL | 0/3 |
| hotpotqa | in-domain | b1t | 90.4 | 91.1 | 91.0 | +0.6 (+0.0, +1.1) | GNX_ABOVE | 3/3 | -0.2 (-0.7, +0.3) | LEVEL | 1/3 |
| hotpotqa | zero-shot | b1w | 92.2 | 92.9 | 93.2 | +1.0 (+0.4, +1.6) | GNX_ABOVE | 3/3 | +0.3 (-0.1, +0.8) | LEVEL | 3/3 |
| hotpotqa | zero-shot | b1t | 88.6 | 89.4 | 89.1 | +0.5 (-0.2, +1.3) | LEVEL | 2/3 | -0.2 (-0.8, +0.3) | LEVEL | 0/3 |

## b1w in-domain against the best published rows

| setting | arm | R@5 (95% interval) | gap to best peer-reviewed | call | gap to best any | call |
|---|---|---|---|---|---|---|
| musique | gnx | 69.1 (67.4, 70.9) | -9.2 | BELOW | -11.6 | BELOW |
| musique | zrckpn | 67.4 (65.7, 69.2) | -10.9 | BELOW | -13.2 | BELOW |
| musique | zg6r | 69.0 (67.2, 70.7) | -9.3 | BELOW | -11.7 | BELOW |
| 2wiki | gnx | 95.0 (94.2, 95.8) | -3.2 | BELOW | -3.5 | BELOW |
| 2wiki | zrckpn | 94.1 (93.2, 94.9) | -4.1 | BELOW | -4.4 | BELOW |
| 2wiki | zg6r | 94.9 (94.1, 95.7) | -3.3 | BELOW | -3.5 | BELOW |
| hotpotqa | gnx | 94.1 (93.1, 95.1) | -3.3 | BELOW | -4.7 | BELOW |
| hotpotqa | zrckpn | 93.4 (92.4, 94.5) | -4.0 | BELOW | -5.3 | BELOW |
| hotpotqa | zg6r | 94.3 (93.3, 95.3) | -3.1 | BELOW | -4.5 | BELOW |
