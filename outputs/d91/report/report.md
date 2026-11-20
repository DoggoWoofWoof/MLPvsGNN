# D91 report: D83 without the labels (the record's three family links), the models of record fitted and read again

Declared in docs/D91_D83_WITHOUT_LABELS.md (section 6, fixed before any read). R@5 x100, candidate p@swa. A cell is the per-question mean over the cell's fits (three seeds; six fits for 2Wiki in-domain), over the questions with a gold, with the 95% interval of the ratio estimator's paired bootstrap over questions (2000 resamples, seed 0): ABOVE, BELOW or LEVEL. In-domain: MuSiQue <- WQ-hotpotqa, HotpotQA <- WQ-musique, 2Wiki <- both; held-out the others.

**Call L (D83 minus D91: the rule names): HELP.** **Call A (D91 minus the record): ADOPT.** D83's own call on the same reads: ADOPT (filed: ADOPT). **Carries: D83.**

## Call L: D83 minus D91 on B1's settings (b1v)

| model | setting | regime | D91 | D83 | D83 - D91 (95%) | call | seed 0 | D83 fits below |
|---|---|---|---:|---:|---|---|---|---:|
| zg6r (GNN) | musique | in-domain | 68.2 | 68.5 | +0.2 (-0.5, +1.0) | LEVEL | -0.1 (-1.1, +1.0) | 2/3 |
| zg6r (GNN) | hotpotqa | in-domain | 94.0 | 94.0 | -0.1 (-0.5, +0.3) | LEVEL | -0.1 (-0.7, +0.5) | 2/3 |
| zg6r (GNN) | 2wiki | in-domain | 95.2 | 95.0 | -0.2 (-0.4, +0.0) | LEVEL | -0.3 (-0.6, +0.0) | 5/6 |
| zg6r (GNN) | musique | held-out | 62.3 | 61.6 | -0.7 (-1.4, +0.1) | LEVEL | -0.6 (-1.7, +0.5) | 3/3 |
| zg6r (GNN) | hotpotqa | held-out | 93.1 | 93.0 | -0.1 (-0.5, +0.3) | LEVEL | -0.5 (-1.1, +0.1) | 1/3 |
| GNX (GNN, same inputs) | musique | in-domain | 68.3 | 68.5 | +0.2 (-0.4, +0.9) | LEVEL | -0.1 (-1.1, +0.8) | 2/3 |
| GNX (GNN, same inputs) | hotpotqa | in-domain | 94.1 | 94.3 | +0.2 (-0.1, +0.7) | LEVEL | -0.1 (-0.7, +0.4) | 1/3 |
| GNX (GNN, same inputs) | 2wiki | in-domain | 95.3 | 95.1 | -0.1 (-0.3, +0.1) | LEVEL | +0.1 (-0.2, +0.3) | 4/6 |
| GNX (GNN, same inputs) | musique | held-out | 63.7 | 64.1 | +0.4 (-0.3, +1.0) | LEVEL | +1.7 (+0.6, +2.8) | 1/3 |
| GNX (GNN, same inputs) | hotpotqa | held-out | 93.0 | 92.9 | -0.1 (-0.5, +0.3) | LEVEL | -0.2 (-0.8, +0.4) | 2/3 |
| zrckpn (MLP) | musique | in-domain | 67.5 | 67.9 | +0.4 (-0.2, +0.9) | LEVEL | +1.2 (+0.4, +2.2) | 1/3 |
| zrckpn (MLP) | hotpotqa | in-domain | 93.7 | 93.6 | -0.0 (-0.4, +0.4) | LEVEL | +0.0 (-0.6, +0.6) | 1/3 |
| zrckpn (MLP) | 2wiki | in-domain | 94.5 | 94.4 | -0.1 (-0.3, +0.2) | LEVEL | -0.1 (-0.4, +0.2) | 4/6 |
| zrckpn (MLP) | musique | held-out | 62.8 | 64.5 | +1.7 (+1.1, +2.3) | ABOVE | +1.5 (+0.7, +2.5) | 0/3 |
| zrckpn (MLP) | hotpotqa | held-out | 92.0 | 92.2 | +0.2 (-0.2, +0.7) | LEVEL | +0.5 (-0.1, +1.2) | 0/3 |

## Call A: D91 minus the record on B1's settings (b1v against b1w)

| model | setting | regime | record | D91 | D91 - record (95%) | call | seed 0 | D91 fits below |
|---|---|---|---:|---:|---|---|---|---:|
| zg6r (GNN) | musique | in-domain | 69.0 | 68.2 | -0.8 (-1.6, +0.0) | LEVEL | -0.5 (-1.6, +0.6) | 3/3 |
| zg6r (GNN) | hotpotqa | in-domain | 94.3 | 94.0 | -0.2 (-0.7, +0.2) | LEVEL | -0.4 (-1.0, +0.2) | 2/3 |
| zg6r (GNN) | 2wiki | in-domain | 94.9 | 95.2 | +0.2 (-0.0, +0.5) | LEVEL | +0.1 (-0.1, +0.4) | 1/6 |
| zg6r (GNN) | musique | held-out | 60.8 | 62.3 | +1.5 (+0.8, +2.3) | ABOVE | +1.5 (+0.4, +2.7) | 0/3 |
| zg6r (GNN) | hotpotqa | held-out | 92.9 | 93.1 | +0.2 (-0.3, +0.7) | LEVEL | +0.1 (-0.5, +0.7) | 0/3 |
| GNX (GNN, same inputs) | musique | in-domain | 69.1 | 68.3 | -0.7 (-1.5, +0.1) | LEVEL | -1.0 (-2.1, +0.0) | 3/3 |
| GNX (GNN, same inputs) | hotpotqa | in-domain | 94.1 | 94.1 | +0.0 (-0.4, +0.4) | LEVEL | +0.1 (-0.5, +0.5) | 1/3 |
| GNX (GNN, same inputs) | 2wiki | in-domain | 95.0 | 95.3 | +0.3 (+0.0, +0.5) | ABOVE | +0.2 (-0.1, +0.4) | 1/6 |
| GNX (GNN, same inputs) | musique | held-out | 63.1 | 63.7 | +0.6 (-0.1, +1.4) | LEVEL | -0.3 (-1.4, +0.9) | 1/3 |
| GNX (GNN, same inputs) | hotpotqa | held-out | 93.2 | 93.0 | -0.2 (-0.8, +0.3) | LEVEL | -0.1 (-0.8, +0.5) | 2/3 |
| zrckpn (MLP) | musique | in-domain | 67.4 | 67.5 | +0.1 (-0.5, +0.7) | LEVEL | -0.7 (-1.6, +0.3) | 1/3 |
| zrckpn (MLP) | hotpotqa | in-domain | 93.4 | 93.7 | +0.2 (-0.1, +0.6) | LEVEL | +0.2 (-0.2, +0.8) | 0/3 |
| zrckpn (MLP) | 2wiki | in-domain | 94.1 | 94.5 | +0.4 (+0.1, +0.7) | ABOVE | +0.2 (-0.2, +0.5) | 0/6 |
| zrckpn (MLP) | musique | held-out | 61.8 | 62.8 | +1.0 (+0.4, +1.6) | ABOVE | +2.5 (+1.5, +3.5) | 0/3 |
| zrckpn (MLP) | hotpotqa | held-out | 92.2 | 92.0 | -0.2 (-0.7, +0.1) | LEVEL | -0.7 (-1.4, +0.0) | 2/3 |

## FC@5 and hit@1 on Call L's cells (beside)

| model | setting | regime | FC@5 D91 | FC@5 D83 | FC@5 D83 - D91 (95%) | hit@1 D91 | hit@1 D83 | hit@1 D83 - D91 (95%) |
|---|---|---|---:|---:|---|---:|---:|---|
| zg6r (GNN) | musique | in-domain | 38.3 | 38.7 | +0.4 (-0.9, +1.8) | 81.5 | 82.0 | +0.5 (-0.4, +1.3) |
| zg6r (GNN) | hotpotqa | in-domain | 88.7 | 88.6 | -0.2 (-0.9, +0.5) | 91.1 | 90.6 | -0.5 (-1.2, +0.1) |
| zg6r (GNN) | 2wiki | in-domain | 87.9 | 87.5 | -0.4 (-0.9, +0.1) | 99.4 | 99.5 | +0.1 (-0.1, +0.2) |
| zg6r (GNN) | musique | held-out | 30.8 | 30.3 | -0.5 (-1.7, +0.8) | 76.9 | 76.3 | -0.6 (-1.7, +0.5) |
| zg6r (GNN) | hotpotqa | held-out | 87.1 | 86.8 | -0.3 (-1.0, +0.4) | 87.0 | 87.6 | +0.6 (-0.1, +1.4) |
| GNX (GNN, same inputs) | musique | in-domain | 38.7 | 38.6 | -0.1 (-1.4, +1.2) | 80.7 | 81.8 | +1.1 (+0.1, +2.1) |
| GNX (GNN, same inputs) | hotpotqa | in-domain | 88.8 | 89.3 | +0.4 (-0.3, +1.2) | 90.9 | 90.8 | -0.0 (-0.5, +0.5) |
| GNX (GNN, same inputs) | 2wiki | in-domain | 88.1 | 87.6 | -0.4 (-0.9, +0.1) | 99.4 | 99.4 | +0.0 (-0.1, +0.2) |
| GNX (GNN, same inputs) | musique | held-out | 32.6 | 32.6 | +0.0 (-1.1, +1.1) | 75.9 | 76.4 | +0.5 (-0.7, +1.6) |
| GNX (GNN, same inputs) | hotpotqa | held-out | 87.1 | 86.9 | -0.3 (-0.9, +0.4) | 87.2 | 86.7 | -0.5 (-1.1, +0.1) |
| zrckpn (MLP) | musique | in-domain | 37.6 | 37.8 | +0.2 (-0.8, +1.2) | 79.8 | 80.5 | +0.7 (-0.1, +1.5) |
| zrckpn (MLP) | hotpotqa | in-domain | 87.9 | 87.8 | -0.0 (-0.7, +0.7) | 90.2 | 90.4 | +0.2 (-0.4, +0.8) |
| zrckpn (MLP) | 2wiki | in-domain | 85.7 | 85.5 | -0.2 (-0.8, +0.3) | 98.9 | 98.7 | -0.2 (-0.5, -0.0) |
| zrckpn (MLP) | musique | held-out | 31.7 | 33.6 | +1.9 (+0.9, +3.1) | 76.7 | 76.2 | -0.4 (-1.4, +0.6) |
| zrckpn (MLP) | hotpotqa | held-out | 84.9 | 85.2 | +0.3 (-0.5, +1.1) | 90.4 | 90.2 | -0.2 (-0.9, +0.4) |

## By hops (MuSiQue) and by kind (2Wiki, HotpotQA), R@5 (beside, descriptive)

| call | model | setting | regime | group | questions | lower side | higher side | difference (95%) |
|---|---|---|---|---|---:|---:|---:|---|
| L | zg6r (GNN) | musique | in-domain | 2 hops | 518 | d91 79.1 | d83 78.9 | -0.2 (-1.3, +1.0) |
| L | zg6r (GNN) | musique | in-domain | 3 hops | 316 | d91 63.2 | d83 63.6 | +0.4 (-0.8, +1.7) |
| L | zg6r (GNN) | musique | in-domain | 4 hops | 166 | d91 43.8 | d83 45.1 | +1.3 (+0.1, +2.4) |
| A | zg6r (GNN) | musique | in-domain | 2 hops | 518 | record 79.0 | d91 79.1 | +0.0 (-1.3, +1.3) |
| A | zg6r (GNN) | musique | in-domain | 3 hops | 316 | record 64.2 | d91 63.2 | -0.9 (-2.2, +0.3) |
| A | zg6r (GNN) | musique | in-domain | 4 hops | 166 | record 46.6 | d91 43.8 | -2.8 (-4.3, -1.3) |
| L | zg6r (GNN) | hotpotqa | in-domain | bridge | 811 | d91 92.7 | d83 92.6 | -0.1 (-0.6, +0.4) |
| L | zg6r (GNN) | hotpotqa | in-domain | comparison | 189 | d91 99.9 | d83 99.9 | +0.0 (+0.0, +0.0) |
| A | zg6r (GNN) | hotpotqa | in-domain | bridge | 811 | record 93.0 | d91 92.7 | -0.3 (-0.8, +0.2) |
| A | zg6r (GNN) | hotpotqa | in-domain | comparison | 189 | record 99.9 | d91 99.9 | +0.0 (+0.0, +0.0) |
| L | zg6r (GNN) | 2wiki | in-domain | compositional | 413 | d91 95.0 | d83 94.9 | -0.1 (-0.6, +0.3) |
| L | zg6r (GNN) | 2wiki | in-domain | inference | 108 | d91 93.9 | d83 93.4 | -0.5 (-1.6, +0.4) |
| L | zg6r (GNN) | 2wiki | in-domain | comparison | 244 | d91 99.9 | d83 100.0 | +0.0 (+0.0, +0.1) |
| L | zg6r (GNN) | 2wiki | in-domain | bridge_comparison | 235 | d91 91.0 | d83 90.7 | -0.3 (-0.7, +0.0) |
| A | zg6r (GNN) | 2wiki | in-domain | compositional | 413 | record 94.7 | d91 95.0 | +0.3 (-0.1, +0.8) |
| A | zg6r (GNN) | 2wiki | in-domain | inference | 108 | record 93.8 | d91 93.9 | +0.1 (-1.0, +1.2) |
| A | zg6r (GNN) | 2wiki | in-domain | comparison | 244 | record 100.0 | d91 99.9 | -0.1 (-0.2, +0.0) |
| A | zg6r (GNN) | 2wiki | in-domain | bridge_comparison | 235 | record 90.5 | d91 91.0 | +0.5 (+0.1, +0.9) |
| L | zg6r (GNN) | musique | held-out | 2 hops | 518 | d91 74.2 | d83 73.2 | -1.0 (-2.1, +0.1) |
| L | zg6r (GNN) | musique | held-out | 3 hops | 316 | d91 55.1 | d83 54.4 | -0.7 (-1.9, +0.6) |
| L | zg6r (GNN) | musique | held-out | 4 hops | 166 | d91 39.0 | d83 39.2 | +0.3 (-1.1, +1.5) |
| A | zg6r (GNN) | musique | held-out | 2 hops | 518 | record 72.0 | d91 74.2 | +2.2 (+1.0, +3.4) |
| A | zg6r (GNN) | musique | held-out | 3 hops | 316 | record 54.0 | d91 55.1 | +1.1 (-0.2, +2.2) |
| A | zg6r (GNN) | musique | held-out | 4 hops | 166 | record 38.7 | d91 39.0 | +0.3 (-1.0, +1.6) |
| L | zg6r (GNN) | hotpotqa | held-out | bridge | 811 | d91 91.5 | d83 91.4 | -0.1 (-0.6, +0.3) |
| L | zg6r (GNN) | hotpotqa | held-out | comparison | 189 | d91 100.0 | d83 100.0 | +0.0 (+0.0, +0.0) |
| A | zg6r (GNN) | hotpotqa | held-out | bridge | 811 | record 91.2 | d91 91.5 | +0.3 (-0.3, +0.9) |
| A | zg6r (GNN) | hotpotqa | held-out | comparison | 189 | record 100.0 | d91 100.0 | +0.0 (+0.0, +0.0) |
| L | GNX (GNN, same inputs) | musique | in-domain | 2 hops | 518 | d91 79.3 | d83 78.9 | -0.5 (-1.5, +0.5) |
| L | GNX (GNN, same inputs) | musique | in-domain | 3 hops | 316 | d91 62.5 | d83 63.6 | +1.1 (+0.0, +2.1) |
| L | GNX (GNN, same inputs) | musique | in-domain | 4 hops | 166 | d91 45.1 | d83 45.8 | +0.8 (-0.3, +1.8) |
| A | GNX (GNN, same inputs) | musique | in-domain | 2 hops | 518 | record 79.2 | d91 79.3 | +0.2 (-1.0, +1.4) |
| A | GNX (GNN, same inputs) | musique | in-domain | 3 hops | 316 | record 64.3 | d91 62.5 | -1.8 (-3.0, -0.7) |
| A | GNX (GNN, same inputs) | musique | in-domain | 4 hops | 166 | record 46.6 | d91 45.1 | -1.6 (-2.9, -0.3) |
| L | GNX (GNN, same inputs) | hotpotqa | in-domain | bridge | 811 | d91 92.7 | d83 93.1 | +0.3 (-0.1, +0.8) |
| L | GNX (GNN, same inputs) | hotpotqa | in-domain | comparison | 189 | d91 99.9 | d83 99.9 | +0.0 (+0.0, +0.0) |
| A | GNX (GNN, same inputs) | hotpotqa | in-domain | bridge | 811 | record 92.7 | d91 92.7 | +0.1 (-0.4, +0.6) |
| A | GNX (GNN, same inputs) | hotpotqa | in-domain | comparison | 189 | record 100.0 | d91 99.9 | -0.1 (-0.3, +0.0) |
| L | GNX (GNN, same inputs) | 2wiki | in-domain | compositional | 413 | d91 95.4 | d83 95.3 | -0.1 (-0.4, +0.3) |
| L | GNX (GNN, same inputs) | 2wiki | in-domain | inference | 108 | d91 94.2 | d83 94.1 | -0.1 (-0.9, +0.7) |
| L | GNX (GNN, same inputs) | 2wiki | in-domain | comparison | 244 | d91 99.9 | d83 99.9 | +0.0 (+0.0, +0.0) |
| L | GNX (GNN, same inputs) | 2wiki | in-domain | bridge_comparison | 235 | d91 90.8 | d83 90.3 | -0.5 (-0.9, -0.1) |
| A | GNX (GNN, same inputs) | 2wiki | in-domain | compositional | 413 | record 95.0 | d91 95.4 | +0.3 (-0.1, +0.8) |
| A | GNX (GNN, same inputs) | 2wiki | in-domain | inference | 108 | record 93.2 | d91 94.2 | +1.0 (+0.1, +1.9) |
| A | GNX (GNN, same inputs) | 2wiki | in-domain | comparison | 244 | record 100.0 | d91 99.9 | -0.0 (-0.1, +0.0) |
| A | GNX (GNN, same inputs) | 2wiki | in-domain | bridge_comparison | 235 | record 90.6 | d91 90.8 | +0.2 (-0.3, +0.6) |
| L | GNX (GNN, same inputs) | musique | held-out | 2 hops | 518 | d91 75.6 | d83 75.6 | +0.1 (-1.0, +1.0) |
| L | GNX (GNN, same inputs) | musique | held-out | 3 hops | 316 | d91 56.3 | d83 57.5 | +1.3 (+0.2, +2.5) |
| L | GNX (GNN, same inputs) | musique | held-out | 4 hops | 166 | d91 40.9 | d83 40.4 | -0.5 (-1.6, +0.6) |
| A | GNX (GNN, same inputs) | musique | held-out | 2 hops | 518 | record 74.5 | d91 75.6 | +1.1 (-0.1, +2.3) |
| A | GNX (GNN, same inputs) | musique | held-out | 3 hops | 316 | record 56.3 | d91 56.3 | -0.1 (-1.3, +1.1) |
| A | GNX (GNN, same inputs) | musique | held-out | 4 hops | 166 | record 40.3 | d91 40.9 | +0.6 (-0.6, +1.9) |
| L | GNX (GNN, same inputs) | hotpotqa | held-out | bridge | 811 | d91 91.4 | d83 91.3 | -0.1 (-0.6, +0.3) |
| L | GNX (GNN, same inputs) | hotpotqa | held-out | comparison | 189 | d91 100.0 | d83 99.9 | -0.1 (-0.3, +0.0) |
| A | GNX (GNN, same inputs) | hotpotqa | held-out | bridge | 811 | record 91.7 | d91 91.4 | -0.3 (-0.9, +0.3) |
| A | GNX (GNN, same inputs) | hotpotqa | held-out | comparison | 189 | record 99.9 | d91 100.0 | +0.1 (+0.0, +0.3) |
| L | zrckpn (MLP) | musique | in-domain | 2 hops | 518 | d91 78.2 | d83 78.8 | +0.5 (-0.3, +1.3) |
| L | zrckpn (MLP) | musique | in-domain | 3 hops | 316 | d91 62.5 | d83 62.6 | +0.1 (-0.9, +1.1) |
| L | zrckpn (MLP) | musique | in-domain | 4 hops | 166 | d91 43.4 | d83 43.8 | +0.4 (-0.5, +1.3) |
| A | zrckpn (MLP) | musique | in-domain | 2 hops | 518 | record 77.9 | d91 78.2 | +0.3 (-0.5, +1.3) |
| A | zrckpn (MLP) | musique | in-domain | 3 hops | 316 | record 62.8 | d91 62.5 | -0.2 (-1.3, +0.8) |
| A | zrckpn (MLP) | musique | in-domain | 4 hops | 166 | record 43.5 | d91 43.4 | -0.1 (-1.0, +0.8) |
| L | zrckpn (MLP) | hotpotqa | in-domain | bridge | 811 | d91 92.2 | d83 92.2 | -0.0 (-0.5, +0.5) |
| L | zrckpn (MLP) | hotpotqa | in-domain | comparison | 189 | d91 99.8 | d83 99.8 | +0.0 (+0.0, +0.0) |
| A | zrckpn (MLP) | hotpotqa | in-domain | bridge | 811 | record 91.9 | d91 92.2 | +0.3 (-0.2, +0.8) |
| A | zrckpn (MLP) | hotpotqa | in-domain | comparison | 189 | record 99.8 | d91 99.8 | +0.0 (+0.0, +0.0) |
| L | zrckpn (MLP) | 2wiki | in-domain | compositional | 413 | d91 94.9 | d83 94.8 | -0.1 (-0.6, +0.3) |
| L | zrckpn (MLP) | 2wiki | in-domain | inference | 108 | d91 92.5 | d83 93.3 | +0.8 (+0.0, +1.5) |
| L | zrckpn (MLP) | 2wiki | in-domain | comparison | 244 | d91 100.0 | d83 100.0 | +0.0 (+0.0, +0.0) |
| L | zrckpn (MLP) | 2wiki | in-domain | bridge_comparison | 235 | d91 88.8 | d83 88.4 | -0.4 (-0.9, +0.0) |
| A | zrckpn (MLP) | 2wiki | in-domain | compositional | 413 | record 94.1 | d91 94.9 | +0.8 (+0.3, +1.4) |
| A | zrckpn (MLP) | 2wiki | in-domain | inference | 108 | record 92.4 | d91 92.5 | +0.2 (-0.8, +1.2) |
| A | zrckpn (MLP) | 2wiki | in-domain | comparison | 244 | record 100.0 | d91 100.0 | +0.0 (+0.0, +0.0) |
| A | zrckpn (MLP) | 2wiki | in-domain | bridge_comparison | 235 | record 88.7 | d91 88.8 | +0.1 (-0.4, +0.6) |
| L | zrckpn (MLP) | musique | held-out | 2 hops | 518 | d91 74.1 | d83 76.0 | +1.9 (+0.9, +2.8) |
| L | zrckpn (MLP) | musique | held-out | 3 hops | 316 | d91 57.1 | d83 58.8 | +1.7 (+0.7, +2.7) |
| L | zrckpn (MLP) | musique | held-out | 4 hops | 166 | d91 38.4 | d83 39.6 | +1.2 (+0.3, +2.2) |
| A | zrckpn (MLP) | musique | held-out | 2 hops | 518 | record 72.7 | d91 74.1 | +1.4 (+0.4, +2.2) |
| A | zrckpn (MLP) | musique | held-out | 3 hops | 316 | record 55.9 | d91 57.1 | +1.2 (+0.2, +2.2) |
| A | zrckpn (MLP) | musique | held-out | 4 hops | 166 | record 39.2 | d91 38.4 | -0.8 (-1.7, +0.1) |
| L | zrckpn (MLP) | hotpotqa | held-out | bridge | 811 | d91 90.1 | d83 90.4 | +0.3 (-0.2, +0.8) |
| L | zrckpn (MLP) | hotpotqa | held-out | comparison | 189 | d91 100.0 | d83 100.0 | +0.0 (+0.0, +0.0) |
| A | zrckpn (MLP) | hotpotqa | held-out | bridge | 811 | record 90.4 | d91 90.1 | -0.3 (-0.8, +0.2) |
| A | zrckpn (MLP) | hotpotqa | held-out | comparison | 189 | record 100.0 | d91 100.0 | +0.0 (+0.0, +0.0) |

## The gaps between the models, at D91 beside D83's (R@5, paired by question)

| carve | dataset | regime | gap | D83 | D91 | change (D91 - D83) |
|---|---|---|---|---|---|---|
| b1v | musique | in-domain | gnx-zrckpn | +0.7 (+0.0, +1.4) | +0.8 (+0.1, +1.6) | +0.2 (-0.7, +1.0) |
| b1v | musique | in-domain | zg6r-zrckpn | +0.6 (-0.2, +1.4) | +0.7 (-0.2, +1.5) | +0.1 (-0.8, +1.0) |
| b1v | hotpotqa | in-domain | gnx-zrckpn | +0.7 (+0.3, +1.2) | +0.4 (+0.0, +0.9) | -0.3 (-0.8, +0.3) |
| b1v | hotpotqa | in-domain | zg6r-zrckpn | +0.3 (-0.1, +0.8) | +0.4 (-0.1, +0.9) | +0.1 (-0.5, +0.6) |
| b1v | 2wiki | in-domain | gnx-zrckpn | +0.7 (+0.5, +1.0) | +0.8 (+0.5, +1.1) | +0.1 (-0.2, +0.3) |
| b1v | 2wiki | in-domain | zg6r-zrckpn | +0.6 (+0.2, +0.9) | +0.7 (+0.3, +1.0) | +0.1 (-0.1, +0.4) |
| b1v | musique | held-out | gnx-zrckpn | -0.5 (-1.2, +0.3) | +0.9 (+0.1, +1.7) | +1.4 (+0.5, +2.2) |
| b1v | musique | held-out | zg6r-zrckpn | -2.9 (-3.8, -1.9) | -0.5 (-1.4, +0.4) | +2.4 (+1.4, +3.4) |
| b1v | hotpotqa | held-out | gnx-zrckpn | +0.7 (+0.1, +1.2) | +1.0 (+0.5, +1.6) | +0.3 (-0.2, +0.8) |
| b1v | hotpotqa | held-out | zg6r-zrckpn | +0.8 (+0.2, +1.3) | +1.1 (+0.6, +1.7) | +0.3 (-0.2, +0.9) |
| s1eval | squad | in-domain | gnx-zrckpn | -0.0 (-0.1, +0.1) | +0.0 (-0.1, +0.1) | +0.0 (-0.1, +0.2) |
| s1eval | squad | in-domain | zg6r-zrckpn | -0.0 (-0.2, +0.1) | -0.0 (-0.2, +0.1) | +0.0 (-0.1, +0.1) |
| s1eval | hotpotqa | in-domain | gnx-zrckpn | +1.3 (+1.1, +1.5) | +1.0 (+0.8, +1.2) | -0.3 (-0.5, -0.0) |
| s1eval | hotpotqa | in-domain | zg6r-zrckpn | +1.2 (+1.0, +1.5) | +0.7 (+0.4, +0.9) | -0.6 (-0.8, -0.3) |
| s1eval | hotpotqa | held-out | gnx-zrckpn | -1.0 (-1.3, -0.7) | -1.2 (-1.4, -0.8) | -0.2 (-0.5, +0.1) |
| s1eval | hotpotqa | held-out | zg6r-zrckpn | -1.8 (-2.2, -1.5) | -2.0 (-2.3, -1.7) | -0.1 (-0.4, +0.2) |
| s1eval | 2wiki | in-domain | gnx-zrckpn | +1.0 (+0.9, +1.1) | +1.1 (+1.0, +1.2) | +0.1 (+0.0, +0.2) |
| s1eval | 2wiki | in-domain | zg6r-zrckpn | +0.9 (+0.8, +1.0) | +0.9 (+0.8, +1.1) | +0.0 (-0.1, +0.2) |
| s1eval | musique | in-domain | gnx-zrckpn | +0.9 (+0.4, +1.4) | +1.0 (+0.6, +1.5) | +0.1 (-0.3, +0.6) |
| s1eval | musique | in-domain | zg6r-zrckpn | +0.9 (+0.4, +1.5) | +1.1 (+0.6, +1.6) | +0.1 (-0.4, +0.6) |
| s1eval | musique | held-out | gnx-zrckpn | -0.9 (-1.4, -0.4) | -0.3 (-0.8, +0.2) | +0.6 (+0.1, +1.1) |
| s1eval | musique | held-out | zg6r-zrckpn | -3.4 (-3.9, -2.8) | -1.1 (-1.6, -0.6) | +2.3 (+1.7, +2.8) |
| s1eval | metaqa | held-out | gnx-zrckpn | -2.3 (-2.8, -1.8) | -1.7 (-2.3, -1.2) | +0.6 (+0.2, +1.0) |
| s1eval | metaqa | held-out | zg6r-zrckpn | -2.7 (-3.2, -2.3) | -1.7 (-2.2, -1.1) | +1.1 (+0.7, +1.5) |
| s1eval | webqsp | in-domain | gnx-zrckpn | +0.5 (-0.7, +1.6) | +0.9 (-0.3, +2.1) | +0.4 (-0.2, +1.0) |
| s1eval | webqsp | in-domain | zg6r-zrckpn | +0.3 (-1.0, +1.5) | +0.6 (-0.6, +1.8) | +0.3 (-0.3, +0.9) |

## The six s1eval carves (beside, deciding nothing)

| model | dataset | regime | questions | D91 | D83 - D91 (95%) | D91 - record (95%) |
|---|---|---|---:|---:|---|---|
| zg6r (GNN) | squad | in-domain | 11873 | 91.3 | +0.1 (+0.0, +0.2) | -0.1 (-0.2, -0.0) |
| zg6r (GNN) | hotpotqa | in-domain | 7405 | 85.4 | +0.4 (+0.2, +0.6) | -0.5 (-0.7, -0.3) |
| zg6r (GNN) | hotpotqa | held-out | 7405 | 79.7 | -0.0 (-0.3, +0.2) | +0.4 (+0.2, +0.6) |
| zg6r (GNN) | 2wiki | in-domain | 12576 | 92.5 | +0.1 (-0.0, +0.1) | -0.2 (-0.3, -0.1) |
| zg6r (GNN) | musique | in-domain | 2417 | 59.0 | +0.4 (-0.0, +0.8) | -0.4 (-0.9, +0.0) |
| zg6r (GNN) | musique | held-out | 2417 | 52.3 | -0.7 (-1.1, -0.3) | +1.4 (+1.0, +1.8) |
| zg6r (GNN) | metaqa | held-out | 9785 | 37.8 | +2.0 (+1.8, +2.3) | +1.8 (+1.5, +2.0) |
| zg6r (GNN) | webqsp | in-domain | 1503 | 65.5 | -0.3 (-0.7, +0.2) | -0.3 (-0.8, +0.1) |
| GNX (GNN, same inputs) | squad | in-domain | 11873 | 91.3 | +0.1 (-0.0, +0.2) | -0.1 (-0.2, +0.0) |
| GNX (GNN, same inputs) | hotpotqa | in-domain | 7405 | 85.7 | +0.1 (-0.0, +0.3) | -0.2 (-0.4, -0.1) |
| GNX (GNN, same inputs) | hotpotqa | held-out | 7405 | 80.5 | +0.0 (-0.2, +0.3) | +0.4 (+0.2, +0.7) |
| GNX (GNN, same inputs) | 2wiki | in-domain | 12576 | 92.7 | -0.0 (-0.1, +0.1) | +0.1 (-0.0, +0.2) |
| GNX (GNN, same inputs) | musique | in-domain | 2417 | 59.0 | +0.4 (-0.0, +0.7) | -0.5 (-0.9, -0.0) |
| GNX (GNN, same inputs) | musique | held-out | 2417 | 53.1 | +1.0 (+0.5, +1.4) | +0.8 (+0.3, +1.2) |
| GNX (GNN, same inputs) | metaqa | held-out | 9785 | 37.8 | +2.5 (+2.2, +2.8) | +0.4 (+0.1, +0.6) |
| GNX (GNN, same inputs) | webqsp | in-domain | 1503 | 65.8 | -0.3 (-0.8, +0.1) | -0.2 (-0.6, +0.3) |
| zrckpn (MLP) | squad | in-domain | 11873 | 91.3 | +0.1 (+0.0, +0.2) | +0.0 (-0.1, +0.1) |
| zrckpn (MLP) | hotpotqa | in-domain | 7405 | 84.7 | -0.1 (-0.3, +0.0) | -0.0 (-0.2, +0.1) |
| zrckpn (MLP) | hotpotqa | held-out | 7405 | 81.7 | -0.2 (-0.4, +0.0) | -0.0 (-0.2, +0.2) |
| zrckpn (MLP) | 2wiki | in-domain | 12576 | 91.6 | +0.1 (+0.0, +0.2) | +0.1 (+0.0, +0.2) |
| zrckpn (MLP) | musique | in-domain | 2417 | 58.0 | +0.5 (+0.2, +0.8) | +0.0 (-0.3, +0.4) |
| zrckpn (MLP) | musique | held-out | 2417 | 53.4 | +1.6 (+1.2, +1.9) | +0.7 (+0.4, +1.0) |
| zrckpn (MLP) | metaqa | held-out | 9785 | 39.5 | +3.1 (+2.7, +3.4) | -1.5 (-1.7, -1.3) |
| zrckpn (MLP) | webqsp | in-domain | 1503 | 64.9 | +0.1 (-0.4, +0.5) | -0.2 (-0.7, +0.1) |

## Links per family (D91) beside links per label (D83): unique pairs per question, gold-touching, gold-gold

**b1v musique**

| store | family or label | links | gold-touching | gold-gold | questions with a gold-gold link |
|---|---|---:|---:|---:|---:|
| D91 | F0 slot (S on a KB, R.mention on passages) | 13351661 | 26512 | 479 | 400 |
| D91 | F1 NER (F0T; minus S on a KB) | 19858405 | 58690 | 1098 | 807 |
| D91 | F2 K.knn | 3155843 | 6705 | 25 | 25 |
| D83 | S | 0 | 0 | 0 | 0 |
| D83 | R.mention | 13353884 | 26512 | 479 | 400 |
| D83 | R.name_comention | 18651908 | 54727 | 1008 | 748 |
| D83 | R.title_comention | 1206545 | 3964 | 90 | 87 |
| D83 | K.knn | 5198733 | 12495 | 355 | 349 |

**b1v hotpotqa**

| store | family or label | links | gold-touching | gold-gold | questions with a gold-gold link |
|---|---|---:|---:|---:|---:|
| D91 | F0 slot (S on a KB, R.mention on passages) | 14721 | 2882 | 575 | 575 |
| D91 | F1 NER (F0T; minus S on a KB) | 109188 | 11502 | 690 | 690 |
| D91 | F2 K.knn | 30806 | 3362 | 45 | 45 |
| D83 | S | 0 | 0 | 0 | 0 |
| D83 | R.mention | 14721 | 2882 | 575 | 575 |
| D83 | R.name_comention | 106970 | 10998 | 581 | 581 |
| D83 | R.title_comention | 2218 | 504 | 109 | 109 |
| D83 | K.knn | 62369 | 7081 | 523 | 523 |

**b1v 2wiki**

| store | family or label | links | gold-touching | gold-gold | questions with a gold-gold link |
|---|---|---:|---:|---:|---:|
| D91 | F0 slot (S on a KB, R.mention on passages) | 16851 | 1880 | 808 | 643 |
| D91 | F1 NER (F0T; minus S on a KB) | 117884 | 9811 | 891 | 707 |
| D91 | F2 K.knn | 44714 | 3667 | 12 | 12 |
| D83 | S | 0 | 0 | 0 | 0 |
| D83 | R.mention | 16851 | 1880 | 808 | 643 |
| D83 | R.name_comention | 115593 | 9510 | 767 | 633 |
| D83 | R.title_comention | 2291 | 301 | 124 | 118 |
| D83 | K.knn | 74448 | 6400 | 579 | 532 |

**s1eval squad**

| store | family or label | links | gold-touching | gold-gold | questions with a gold-gold link |
|---|---|---:|---:|---:|---:|
| D91 | F0 slot (S on a KB, R.mention on passages) | 3051424 | 171563 | 0 | 0 |
| D91 | F1 NER (F0T; minus S on a KB) | 680525 | 46314 | 0 | 0 |
| D91 | F2 K.knn | 75721 | 5650 | 0 | 0 |
| D83 | S | 0 | 0 | 0 | 0 |
| D83 | R.mention | 3051424 | 171563 | 0 | 0 |
| D83 | R.name_comention | 668679 | 45581 | 0 | 0 |
| D83 | R.title_comention | 11846 | 733 | 0 | 0 |
| D83 | K.knn | 490470 | 34983 | 0 | 0 |

**s1eval hotpotqa**

| store | family or label | links | gold-touching | gold-gold | questions with a gold-gold link |
|---|---|---:|---:|---:|---:|
| D91 | F0 slot (S on a KB, R.mention on passages) | 9121020 | 287227 | 4240 | 4240 |
| D91 | F1 NER (F0T; minus S on a KB) | 5470483 | 162769 | 2721 | 2721 |
| D91 | F2 K.knn | 2359894 | 42149 | 193 | 193 |
| D83 | S | 0 | 0 | 0 | 0 |
| D83 | R.mention | 9121121 | 287237 | 4240 | 4240 |
| D83 | R.name_comention | 5183524 | 148122 | 1905 | 1905 |
| D83 | R.title_comention | 286959 | 14647 | 816 | 816 |
| D83 | K.knn | 3606137 | 76236 | 1466 | 1466 |

**s1eval 2wiki**

| store | family or label | links | gold-touching | gold-gold | questions with a gold-gold link |
|---|---|---:|---:|---:|---:|
| D91 | F0 slot (S on a KB, R.mention on passages) | 25974694 | 628948 | 10716 | 8474 |
| D91 | F1 NER (F0T; minus S on a KB) | 16343591 | 538770 | 6849 | 6287 |
| D91 | F2 K.knn | 4221232 | 79718 | 105 | 105 |
| D83 | S | 0 | 0 | 0 | 0 |
| D83 | R.mention | 25975007 | 628965 | 10716 | 8474 |
| D83 | R.name_comention | 15743744 | 516706 | 5830 | 5424 |
| D83 | R.title_comention | 599902 | 22064 | 1019 | 1002 |
| D83 | K.knn | 6985181 | 134294 | 2800 | 2757 |

**s1eval musique**

| store | family or label | links | gold-touching | gold-gold | questions with a gold-gold link |
|---|---|---:|---:|---:|---:|
| D91 | F0 slot (S on a KB, R.mention on passages) | 166513397 | 273524 | 1020 | 889 |
| D91 | F1 NER (F0T; minus S on a KB) | 46673530 | 115285 | 1378 | 1327 |
| D91 | F2 K.knn | 8152497 | 15321 | 29 | 29 |
| D83 | S | 0 | 0 | 0 | 0 |
| D83 | R.mention | 167022173 | 276058 | 1026 | 892 |
| D83 | R.name_comention | 44413984 | 110571 | 1267 | 1223 |
| D83 | R.title_comention | 2259862 | 4714 | 111 | 111 |
| D83 | K.knn | 15213023 | 28745 | 430 | 425 |

**s1eval metaqa**

| store | family or label | links | gold-touching | gold-gold | questions with a gold-gold link |
|---|---|---:|---:|---:|---:|
| D91 | F0 slot (S on a KB, R.mention on passages) | 119272476 | 3729324 | 587 | 423 |
| D91 | F1 NER (F0T; minus S on a KB) | 4160123 | 6139 | 343 | 131 |
| D91 | F2 K.knn | 29817186 | 119374 | 11654 | 1456 |
| D83 | S | 119272476 | 3729324 | 587 | 423 |
| D83 | R.mention | 0 | 0 | 0 | 0 |
| D83 | R.name_comention | 4190038 | 7479 | 343 | 131 |
| D83 | R.title_comention | 0 | 0 | 0 | 0 |
| D83 | K.knn | 31261529 | 123429 | 11859 | 1496 |

**s1eval webqsp**

| store | family or label | links | gold-touching | gold-gold | questions with a gold-gold link |
|---|---|---:|---:|---:|---:|
| D91 | F0 slot (S on a KB, R.mention on passages) | 14697542 | 466823 | 1426 | 228 |
| D91 | F1 NER (F0T; minus S on a KB) | 840378 | 5020 | 1109 | 45 |
| D91 | F2 K.knn | 5971032 | 14615 | 1366 | 228 |
| D83 | S | 14697542 | 466823 | 1426 | 228 |
| D83 | R.mention | 0 | 0 | 0 | 0 |
| D83 | R.name_comention | 979487 | 7402 | 1342 | 60 |
| D83 | R.title_comention | 1177 | 5 | 0 | 0 |
| D83 | K.knn | 6663039 | 22650 | 2040 | 284 |


Missing reads: 0; G6 problems: 0.
