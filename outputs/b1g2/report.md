# B1G Part D2 report: the controls for the oracle edges (declared in docs/SWP_S_TO_R_SWAP.md, section 5d)

R@5 x100 on HippoRAG 2's MuSiQue setting (1,000 questions), no new fit; the models of record (zg6r, zrckpn) read on the rule's W2_S pairs (b1w), on those plus the oracle chain edges (b1c: R + ORC (the chain's dependency edges)), and on two controls built from ORC with the labels but without the right connection: **b1d** (R + OXC (degree control: every gold paragraph keeps its ORC degree, each pair goes to a node that is gold for no question)) and **b1e** (R + OXG (connection control: ORC's pairs with their far ends exchanged between questions, every node keeps its ORC degree)). If a control lifts as much as b1c, the oracle's lift is not the chain's connection. The oracle and the controls use the labels: ceilings and diagnostics, never a method and never an estimate of a graph anyone can build. In-domain = the mean over the fits that trained on MuSiQue's partner split; 95% bootstrap intervals over the questions (2,000 resamples, seed 0); calls by the interval (UP / DOWN / FLAT), no threshold.

## In-domain

| model | b1w | b1c | b1d | b1e | b1c - b1w | b1d - b1w | b1e - b1w | b1c - b1d | b1c - b1e |
|---|---:|---:|---:|---:|---|---|---|---|---|
| zg6r | 69.0 (67.2-70.7) | 79.6 (78.0-81.3) | 67.5 (65.7-69.2) | 67.2 (65.5-69.0) | +10.7 (+9.6, +11.7) | -1.5 (-2.0, -0.9) | -1.7 (-2.2, -1.2) | +12.1 (+11.0, +13.3) | +12.4 (+11.3, +13.5) |
| zrckpn | 67.4 (65.7-69.2) | 78.2 (76.5-79.9) | 66.4 (64.8-68.1) | 66.4 (64.7-68.2) | +10.8 (+9.7, +11.8) | -1.0 (-1.5, -0.5) | -1.0 (-1.5, -0.5) | +11.8 (+10.7, +12.9) | +11.8 (+10.7, +13.0) |

## Held-out (fits that did not train on MuSiQue)

| model | b1w | b1c | b1d | b1e | b1c - b1w | b1d - b1w | b1e - b1w | b1c - b1d | b1c - b1e |
|---|---:|---:|---:|---:|---|---|---|---|---|
| zg6r | 60.8 (59.1-62.5) | 77.0 (75.3-78.8) | 60.5 (58.8-62.2) | 59.9 (58.2-61.6) | +16.3 (+15.0, +17.6) | -0.3 (-1.0, +0.4) | -0.9 (-1.6, -0.3) | +16.6 (+15.3, +17.8) | +17.2 (+15.9, +18.5) |
| zrckpn | 61.8 (60.1-63.6) | 77.0 (75.3-78.8) | 61.8 (60.2-63.6) | 62.1 (60.4-63.9) | +15.2 (+14.0, +16.5) | +0.0 (-0.7, +0.7) | +0.3 (-0.4, +0.9) | +15.2 (+13.9, +16.5) | +14.9 (+13.7, +16.2) |

## By hop depth (in-domain, paired by question)

| hops | questions | model | b1c - b1w | b1d - b1w | b1e - b1w | b1c - b1d | b1c - b1e |
|---|---:|---|---|---|---|---|---|
| 2hop | 518 | zg6r | +7.6 (+6.0, +9.2) | -0.8 (-1.6, +0.0) | -1.1 (-1.9, -0.4) | +8.4 (+6.8, +10.1) | +8.7 (+7.2, +10.4) |
| 2hop | 518 | zrckpn | +8.8 (+7.2, +10.4) | -0.6 (-1.4, +0.0) | -0.5 (-1.3, +0.2) | +9.4 (+7.8, +11.1) | +9.3 (+7.7, +11.0) |
| 3hop | 316 | zg6r | +15.3 (+13.3, +17.2) | -1.9 (-2.8, -1.1) | -2.5 (-3.4, -1.4) | +17.2 (+15.3, +19.3) | +17.8 (+15.6, +19.8) |
| 3hop | 316 | zrckpn | +13.1 (+11.0, +15.1) | -1.2 (-2.0, -0.3) | -1.3 (-2.2, -0.4) | +14.2 (+12.2, +16.4) | +14.3 (+12.2, +16.4) |
| 4hop | 166 | zg6r | +11.5 (+9.4, +13.7) | -2.7 (-3.9, -1.6) | -2.3 (-3.4, -1.1) | +14.2 (+11.8, +16.6) | +13.8 (+11.5, +16.1) |
| 4hop | 166 | zrckpn | +12.6 (+10.4, +14.9) | -1.8 (-2.9, -0.8) | -2.0 (-3.0, -1.0) | +14.4 (+12.0, +16.7) | +14.6 (+12.2, +16.9) |

## The call (in-domain, both models)

| pair | call |
|---|---|
| b1c-b1w | UP |
| b1d-b1w | DOWN |
| b1e-b1w | DOWN |
| b1c-b1d | UP |
| b1c-b1e | UP |
| b1d-b1e | FLAT |

## The control pairs

| carve | pairs | already in W2_S | new | union pairs | directed edges in the file |
|---|---:|---:|---:|---:|---:|
| b1c | 1090 | 309 | 781 | 42210 | 84420 |
| b1d | 2180 | 2 | 2178 | 43607 | 87214 |
| b1e | 1090 | 1 | 1089 | 42518 | 85036 |

Missing reads: 0.
