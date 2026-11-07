# IDPB: the same-input pair on the ladder b1t -> b1w -> b1 (none, rule links, the dataset's own S)

Declared in docs/SWP_S_TO_R_SWAP.md. R@5 in points on HippoRAG 2's settings (1,000 questions each), the 3-seed mean over the splits of the domain, 95% interval of the questions (2000 resamples, seed 0). b1t: structural slot empty, NER and kNN kept. b1w: the rule's links (W2_S) in the slot. b1: the dataset's own S (2wiki, hotpotqa: corpus hyperlinks; musique: title mentions, itself a rule). The fits were trained with the rule's links in the slot, so every b1 row is a read-time swap of the slot, not a model trained on S. No threshold, no adoption call.

The share of the b1 gain at b1w is the ratio of two differences; it is printed only where b1 - b1w is not level (its interval excludes 0), because with overlapping intervals the ratio can pass 100% by noise (the JSON keeps every value).

| setting | domain | model | b1t | b1w | b1 | b1w - b1t | b1 - b1w | b1 - b1w call | b1 - b1t | share of the b1 gain at b1w |
|---|---|---|---:|---:|---:|---|---|---|---|---:|
| 2wiki | in-domain | gnx | 77.6 | 95.0 | 95.3 | +17.4 (+16.1, +18.8) | +0.3 (-0.4, +1.0) | level | +17.8 (+16.4, +19.1) | -- |
| 2wiki | in-domain | zrckpn | 77.0 | 94.1 | 94.4 | +17.1 (+15.8, +18.4) | +0.3 (-0.4, +0.9) | level | +17.4 (+16.1, +18.7) | -- |
| 2wiki | in-domain | zg6r | 78.1 | 94.9 | 95.4 | +16.8 (+15.5, +18.2) | +0.5 (-0.3, +1.1) | level | +17.3 (+16.0, +18.6) | -- |
| hotpotqa | in-domain | gnx | 91.0 | 94.1 | 96.5 | +3.1 (+2.3, +4.0) | +2.4 (+1.7, +3.1) | S above R | +5.5 (+4.6, +6.5) | 0.56 |
| hotpotqa | in-domain | zrckpn | 90.4 | 93.4 | 95.6 | +3.0 (+2.2, +3.9) | +2.2 (+1.5, +2.9) | S above R | +5.2 (+4.3, +6.2) | 0.58 |
| hotpotqa | in-domain | zg6r | 91.1 | 94.3 | 96.5 | +3.1 (+2.4, +4.0) | +2.2 (+1.5, +2.9) | S above R | +5.3 (+4.4, +6.2) | 0.59 |
| hotpotqa | zero-shot | gnx | 89.1 | 93.2 | 94.8 | +4.1 (+3.2, +5.0) | +1.6 (+0.9, +2.2) | S above R | +5.7 (+4.7, +6.7) | 0.72 |
| hotpotqa | zero-shot | zrckpn | 88.6 | 92.2 | 93.9 | +3.6 (+2.8, +4.5) | +1.6 (+0.9, +2.3) | S above R | +5.3 (+4.4, +6.2) | 0.69 |
| hotpotqa | zero-shot | zg6r | 89.4 | 92.9 | 94.7 | +3.5 (+2.6, +4.4) | +1.8 (+1.1, +2.5) | S above R | +5.3 (+4.3, +6.3) | 0.66 |
| musique | in-domain | gnx | 66.2 | 69.1 | 69.0 | +2.8 (+1.7, +3.9) | -0.1 (-0.6, +0.4) | level | +2.8 (+1.7, +3.8) | -- |
| musique | in-domain | zrckpn | 64.5 | 67.4 | 67.3 | +2.9 (+1.9, +4.0) | -0.1 (-0.6, +0.4) | level | +2.9 (+1.8, +4.0) | -- |
| musique | in-domain | zg6r | 66.0 | 69.0 | 68.6 | +3.0 (+1.8, +4.0) | -0.4 (-0.9, +0.2) | level | +2.6 (+1.6, +3.7) | -- |
| musique | zero-shot | gnx | 63.0 | 63.1 | 61.2 | +0.1 (-1.2, +1.4) | -1.9 (-2.6, -1.3) | S below R | -1.8 (-3.1, -0.5) | -- |
| musique | zero-shot | zrckpn | 60.0 | 61.8 | 61.1 | +1.8 (+0.6, +3.1) | -0.8 (-1.3, -0.1) | S below R | +1.1 (-0.1, +2.3) | 1.71 |
| musique | zero-shot | zg6r | 62.1 | 60.8 | 59.1 | -1.3 (-2.6, +0.1) | -1.6 (-2.3, -1.0) | S below R | -2.9 (-4.3, -1.6) | -- |

## Does message passing matter more with structure in? GNX minus zrckpn, same inputs

| setting | domain | b1t | b1w | b1 | gap change b1t -> b1w | gap change b1t -> b1 |
|---|---|---|---|---|---|---|
| 2wiki | in-domain | +0.6 (+0.2, +1.0) GNX_ABOVE | +0.9 (+0.6, +1.2) GNX_ABOVE | +0.9 (+0.6, +1.3) GNX_ABOVE | +0.3 (-0.2, +0.8) | +0.4 (-0.1, +0.8) |
| hotpotqa | in-domain | +0.6 (+0.0, +1.1) GNX_ABOVE | +0.7 (+0.2, +1.1) GNX_ABOVE | +0.9 (+0.5, +1.3) GNX_ABOVE | +0.1 (-0.5, +0.7) | +0.3 (-0.3, +0.9) |
| hotpotqa | zero-shot | +0.5 (-0.2, +1.3) LEVEL | +1.0 (+0.4, +1.6) GNX_ABOVE | +0.9 (+0.4, +1.5) GNX_ABOVE | +0.5 (-0.3, +1.2) | +0.4 (-0.4, +1.2) |
| musique | in-domain | +1.8 (+1.0, +2.5) GNX_ABOVE | +1.6 (+0.9, +2.4) GNX_ABOVE | +1.7 (+0.9, +2.5) GNX_ABOVE | -0.1 (-1.1, +0.8) | -0.1 (-1.1, +0.9) |
| musique | zero-shot | +3.0 (+2.1, +3.8) GNX_ABOVE | +1.2 (+0.4, +2.1) GNX_ABOVE | +0.1 (-0.7, +0.9) LEVEL | -1.7 (-2.7, -0.8) | -2.9 (-3.8, -1.9) |
