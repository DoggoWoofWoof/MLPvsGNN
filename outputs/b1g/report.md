# B1G report (declared in docs/SWP_S_TO_R_SWAP.md, section 5b)

R@5 x100 on HippoRAG 2's MuSiQue setting (1,000 questions), no new fit. The models of record (zg6r, zrckpn; fitted without LLM links, on W2_S pools) are read on b1t (the text links only), b1w (the rule's W2_S links) and two carves whose structural family is **the rule's pairs plus oracle pairs drawn between the supporting paragraphs of each question** (b1c: R + ORC (the chain's dependency edges); b1q: R + ORA (every two supporting paragraphs)). The oracle uses the labels: it is a ceiling for what a better multi-hop graph could give, never a method and never an estimate of a graph anyone can build. In-domain = the mean over the fits that trained on MuSiQue's partner split; 95% bootstrap intervals over the questions (2,000 resamples, seed 0); calls by the interval (UP / DOWN / FLAT), no threshold.

## In-domain

| model | b1t | b1w | b1c | b1q | b1c - b1w | b1q - b1w | b1q - b1c | b1w - b1t |
|---|---:|---:|---:|---:|---|---|---|---|
| zg6r | 66.0 (64.3-67.7) | 69.0 (67.2-70.7) | 79.6 (78.0-81.3) | 80.3 (78.6-81.8) | +10.7 (+9.6, +11.7) | +11.3 (+10.2, +12.5) | +0.6 (+0.1, +1.2) | +3.0 (+1.8, +4.0) |
| zrckpn | 64.5 (62.8-66.1) | 67.4 (65.7-69.2) | 78.2 (76.5-79.9) | 78.9 (77.2-80.5) | +10.8 (+9.7, +11.8) | +11.4 (+10.4, +12.6) | +0.7 (+0.1, +1.2) | +2.9 (+1.9, +4.0) |

## Held-out (fits that did not train on MuSiQue)

| model | b1t | b1w | b1c | b1q | b1c - b1w | b1q - b1w | b1q - b1c | b1w - b1t |
|---|---:|---:|---:|---:|---|---|---|---|
| zg6r | 62.1 (60.4-63.7) | 60.8 (59.1-62.5) | 77.0 (75.3-78.8) | 82.1 (80.5-83.7) | +16.3 (+15.0, +17.6) | +21.3 (+19.9, +22.8) | +5.0 (+4.3, +5.8) | -1.3 (-2.6, +0.1) |
| zrckpn | 60.0 (58.4-61.7) | 61.8 (60.1-63.6) | 77.0 (75.3-78.8) | 81.0 (79.4-82.6) | +15.2 (+14.0, +16.5) | +19.1 (+17.8, +20.6) | +4.0 (+3.3, +4.7) | +1.8 (+0.5, +3.1) |

## Reference: B1L's b1o (H4o's LLM-built P pairs), the filed rows, not a cell of this stage

| model | in-domain b1o | held-out b1o |
|---|---:|---:|
| zg6r | 67.4 (65.8-69.3) | 62.8 (61.1-64.5) |
| zrckpn | 67.3 (65.6-69.2) | 64.0 (62.3-65.8) |

Source `outputs/b1l/report.json` (sha256 acf2956d8d6f). The comparison with b1c / b1q is across stages: same fits, same questions, different link sets.

## By hop depth (in-domain, paired by question)

| hops | questions | model | b1c - b1w | b1q - b1w | b1q - b1c | calls (c-w / q-w / q-c) |
|---|---:|---|---|---|---|---|
| 2hop | 518 | zg6r | +7.6 (+6.0, +9.2) | +7.6 (+6.1, +9.2) | +0.0 (-0.1, +0.2) | UP / UP / FLAT |
| 2hop | 518 | zrckpn | +8.8 (+7.2, +10.4) | +8.8 (+7.2, +10.4) | -0.0 (-0.2, +0.1) | UP / UP / FLAT |
| 3hop | 316 | zg6r | +15.3 (+13.3, +17.2) | +14.4 (+12.3, +16.5) | -0.9 (-2.2, +0.4) | UP / UP / FLAT |
| 3hop | 316 | zrckpn | +13.1 (+11.0, +15.1) | +12.7 (+10.6, +14.7) | -0.4 (-1.7, +0.9) | UP / UP / FLAT |
| 4hop | 166 | zg6r | +11.5 (+9.4, +13.7) | +16.9 (+13.9, +20.1) | +5.4 (+3.1, +7.8) | UP / UP / UP |
| 4hop | 166 | zrckpn | +12.6 (+10.4, +14.9) | +17.5 (+14.6, +20.4) | +4.9 (+2.9, +7.0) | UP / UP / UP |

## The call (in-domain, both models)

| pair | call |
|---|---|
| b1c-b1w | UP |
| b1q-b1w | UP |
| b1q-b1c | UP |
| b1c-b1t | UP |
| b1q-b1t | UP |
| b1w-b1t | UP |

## The oracle pairs

| carve | oracle pairs | already in W2_S | new | union pairs | directed edges in the file |
|---|---:|---:|---:|---:|---:|
| b1c | 1090 | 309 | 781 | 42210 | 84420 |
| b1q | 1750 | 313 | 1437 | 42866 | 85732 |

Missing reads: 0.
