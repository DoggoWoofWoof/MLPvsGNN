# B1b: our trained models on HippoRAG 2's settings (R@5 x100)

| setting | read | R@5 [95% CI] | vs HippoRAG 2 | vs NV-Embed-v2 | gain over RRF [95% CI] | FC@5 | hit@1 |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| musique | our RRF | 56.4 [54.8, 58.2] | | | | 21.3 | 71.4 |
| musique | zsp/J5 (in-domain) | 66.3 [64.5, 68.1] | BELOW (74.7) | BELOW (69.7) | +9.9 [8.3, 11.3] | 36.8 | 77.3 |
| musique | zsp/L-musique (zero-shot) | 60.2 [58.4, 62.2] | BELOW (74.7) | BELOW (69.7) | +3.8 [2.2, 5.4] | 28.9 | 69.7 |
| musique | zrc/J5 (in-domain) | 66.1 [64.3, 67.9] | BELOW (74.7) | BELOW (69.7) | +9.7 [8.2, 11.1] | 35.8 | 77.7 |
| musique | zrc/L-musique (zero-shot) | 61.3 [59.5, 63.2] | BELOW (74.7) | BELOW (69.7) | +4.9 [3.4, 6.4] | 30.4 | 73.2 |
| 2wiki | our RRF | 71.2 [69.7, 72.7] | | | | 41.2 | 97.1 |
| 2wiki | zsp/J5 (in-domain) | 97.2 [96.5, 97.8] | ABOVE (90.4) | ABOVE (76.5) | +25.9 [24.5, 27.3] | 92.6 | 97.5 |
| 2wiki | zsp/L-2wiki (zero-shot) | 93.4 [92.5, 94.3] | ABOVE (90.4) | ABOVE (76.5) | +22.2 [20.8, 23.5] | 79.8 | 97.3 |
| 2wiki | zrc/J5 (in-domain) | 96.4 [95.7, 97.0] | ABOVE (90.4) | ABOVE (76.5) | +25.1 [23.7, 26.6] | 90.0 | 96.8 |
| 2wiki | zrc/L-2wiki (zero-shot) | 93.3 [92.4, 94.1] | ABOVE (90.4) | ABOVE (76.5) | +22.1 [20.7, 23.5] | 79.2 | 97.0 |
| hotpotqa | our RRF | 85.0 [83.6, 86.6] | | | | 71.3 | 88.9 |
| hotpotqa | zsp/J5 (in-domain) | 98.2 [97.6, 98.9] | ABOVE (96.3) | ABOVE (94.5) | +13.2 [11.8, 14.6] | 97.0 | 91.6 |
| hotpotqa | zsp/L-hotpotqa (zero-shot) | 95.9 [95.0, 96.8] | AT (96.3) | ABOVE (94.5) | +10.8 [9.4, 12.1] | 92.2 | 90.1 |
| hotpotqa | zrc/J5 (in-domain) | 98.2 [97.5, 98.8] | ABOVE (96.3) | ABOVE (94.5) | +13.1 [11.7, 14.4] | 96.7 | 90.9 |
| hotpotqa | zrc/L-hotpotqa (zero-shot) | 96.5 [95.7, 97.4] | AT (96.3) | ABOVE (94.5) | +11.5 [10.1, 12.8] | 93.6 | 90.0 |

| setting | pool: golds in pool | pool: all golds in pool | B1d universal | MLP share of GNN gain (J5) | E1: our lift / HippoRAG 2's |
| --- | ---: | ---: | ---: | ---: | --- |
| musique | 98.2 | 95.3 | 57.0 | 0.9838 | 9.8 / 5.0 |
| 2wiki | 98.8 | 96.8 | 92.8 | 0.9699 | 25.9 / 13.9 |
| hotpotqa | 100.0 | 100.0 | 95.2 | 0.9924 | 13.2 / 1.8 |
