# U1a: B1 without the corpus hyperlinks (docs/U1A_B1_WITHOUT_HYPERLINKS.md)

R@5 x100 of the p@swa candidate, 95% bootstrap intervals; b1 = B1b's graph (hyperlinks on 2wiki and hotpotqa), b1t = text only (ner + knn), b1u = text + the title-mention rule.

## The rule against the hyperlinks

| setting | b1 edges | b1u edges | P (dir) | R (dir) | P (undir) | R (undir) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| musique | 45568 | 45586 | identity: 0 pairs differ | | | |
| 2wiki | 2200 | 2346 | 0.7579 | 0.8082 | 0.7636 | 0.8432 |
| hotpotqa | 4856 | 5858 | 0.6574 | 0.793 | 0.6494 | 0.8173 |

## Reads

| setting | read | b1 | b1t | b1u | HippoRAG 2 |
| --- | --- | --- | --- | --- | ---: |
| musique | rrf | 56.4 | 56.4 | 56.4 | 74.7 |
| musique | zrc/J5 | 66.1 [64.3, 67.9] BELOW | 63.4 [61.7, 65.2] BELOW | 66.1 [64.3, 67.9] BELOW | 74.7 |
| musique | zrc/L-musique | 61.3 [59.5, 63.2] BELOW | 58.4 [56.7, 60.0] BELOW | 61.3 [59.5, 63.2] BELOW | 74.7 |
| musique | zsp/J5 | 66.3 [64.5, 68.1] BELOW | 62.9 [61.2, 64.8] BELOW | 66.3 [64.5, 68.1] BELOW | 74.7 |
| musique | zsp/L-musique | 60.2 [58.4, 62.2] BELOW | 57.8 [56.0, 59.4] BELOW | 60.2 [58.4, 62.2] BELOW | 74.7 |
| 2wiki | rrf | 71.2 | 71.2 | 71.2 | 90.4 |
| 2wiki | zrc/J5 | 96.4 [95.7, 97.0] ABOVE | 75.6 [74.1, 77.1] BELOW | 95.3 [94.5, 96.1] ABOVE | 90.4 |
| 2wiki | zrc/L-2wiki | 93.3 [92.4, 94.1] ABOVE | 75.6 [74.1, 77.0] BELOW | 92.2 [91.2, 93.1] ABOVE | 90.4 |
| 2wiki | zsp/J5 | 97.2 [96.5, 97.8] ABOVE | 77.3 [75.8, 78.8] BELOW | 95.9 [95.1, 96.6] ABOVE | 90.4 |
| 2wiki | zsp/L-2wiki | 93.4 [92.5, 94.3] ABOVE | 77.1 [75.6, 78.6] BELOW | 92.2 [91.2, 93.2] ABOVE | 90.4 |
| hotpotqa | rrf | 85.0 | 85.0 | 85.0 | 96.3 |
| hotpotqa | zrc/J5 | 98.2 [97.5, 98.8] ABOVE | 88.7 [87.5, 90.0] BELOW | 94.8 [93.8, 95.8] BELOW | 96.3 |
| hotpotqa | zrc/L-hotpotqa | 96.5 [95.7, 97.4] AT | 88.5 [87.2, 89.9] BELOW | 94.2 [93.2, 95.3] BELOW | 96.3 |
| hotpotqa | zsp/J5 | 98.2 [97.6, 98.9] ABOVE | 90.0 [88.8, 91.2] BELOW | 95.0 [94.0, 96.0] BELOW | 96.3 |
| hotpotqa | zsp/L-hotpotqa | 95.9 [95.0, 96.8] AT | 89.1 [87.8, 90.5] BELOW | 93.8 [92.7, 94.8] BELOW | 96.3 |

## Paired differences (R@5 x100)

| setting | difference | mean | 95% interval |
| --- | --- | ---: | --- |
| musique | b1t-b1 zsp/J5 | -3.3 | [-4.5, -2.1] |
| musique | b1t-b1 zsp/L-musique | -2.5 | [-3.9, -1.0] |
| musique | b1t-b1 zrc/J5 | -2.7 | [-3.8, -1.6] |
| musique | b1t-b1 zrc/L-musique | -2.9 | [-4.3, -1.6] |
| musique | b1u-b1 zsp/J5 | +0.0 | [+0.0, +0.0] |
| musique | b1u-b1 zsp/L-musique | +0.0 | [+0.0, +0.0] |
| musique | b1u-b1 zrc/J5 | +0.0 | [+0.0, +0.0] |
| musique | b1u-b1 zrc/L-musique | +0.0 | [+0.0, +0.0] |
| musique | b1u-b1t zsp/J5 | +3.3 | [+2.1, +4.5] |
| musique | b1u-b1t zsp/L-musique | +2.5 | [+1.0, +3.9] |
| musique | b1u-b1t zrc/J5 | +2.7 | [+1.6, +3.8] |
| musique | b1u-b1t zrc/L-musique | +2.9 | [+1.6, +4.3] |
| 2wiki | b1t-b1 zsp/J5 | -19.8 | [-21.2, -18.3] |
| 2wiki | b1t-b1 zsp/L-2wiki | -16.3 | [-17.6, -15.0] |
| 2wiki | b1t-b1 zrc/J5 | -20.8 | [-22.2, -19.3] |
| 2wiki | b1t-b1 zrc/L-2wiki | -17.7 | [-19.1, -16.4] |
| 2wiki | b1u-b1 zsp/J5 | -1.3 | [-2.1, -0.5] |
| 2wiki | b1u-b1 zsp/L-2wiki | -1.2 | [-1.9, -0.4] |
| 2wiki | b1u-b1 zrc/J5 | -1.1 | [-1.9, -0.2] |
| 2wiki | b1u-b1 zrc/L-2wiki | -1.1 | [-1.8, -0.4] |
| 2wiki | b1u-b1t zsp/J5 | +18.5 | [+17.1, +20.0] |
| 2wiki | b1u-b1t zsp/L-2wiki | +15.2 | [+13.8, +16.5] |
| 2wiki | b1u-b1t zrc/J5 | +19.8 | [+18.3, +21.2] |
| 2wiki | b1u-b1t zrc/L-2wiki | +16.6 | [+15.3, +18.0] |
| hotpotqa | b1t-b1 zsp/J5 | -8.2 | [-9.4, -7.0] |
| hotpotqa | b1t-b1 zsp/L-hotpotqa | -6.8 | [-7.9, -5.7] |
| hotpotqa | b1t-b1 zrc/J5 | -9.4 | [-10.7, -8.2] |
| hotpotqa | b1t-b1 zrc/L-hotpotqa | -8.0 | [-9.2, -6.8] |
| hotpotqa | b1u-b1 zsp/J5 | -3.3 | [-4.1, -2.5] |
| hotpotqa | b1u-b1 zsp/L-hotpotqa | -2.1 | [-2.9, -1.4] |
| hotpotqa | b1u-b1 zrc/J5 | -3.4 | [-4.2, -2.5] |
| hotpotqa | b1u-b1 zrc/L-hotpotqa | -2.3 | [-3.1, -1.5] |
| hotpotqa | b1u-b1t zsp/J5 | +5.0 | [+3.9, +6.0] |
| hotpotqa | b1u-b1t zsp/L-hotpotqa | +4.7 | [+3.6, +5.7] |
| hotpotqa | b1u-b1t zrc/J5 | +6.1 | [+5.0, +7.2] |
| hotpotqa | b1u-b1t zrc/L-hotpotqa | +5.7 | [+4.6, +6.8] |
