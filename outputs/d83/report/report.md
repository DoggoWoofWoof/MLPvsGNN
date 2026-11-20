# D83 report: the unified R with labelled links, the models of record fitted and read again

Declared in docs/D83_UNIFIED_R_REFIT.md (section 7, fixed before any read). R@5 x100, candidate p@swa. A cell is D83 minus the record on the per-question mean over the cell's fits (three seeds; six fits for 2Wiki in-domain), over the questions with a gold, with the 95% interval of the ratio estimator's paired bootstrap over questions (2000 resamples, seed 0): ABOVE, BELOW or LEVEL. The record: zg6r on b1w (B1P), zrckpn on b1w (B1M), GNX on b1w (IDPA); on s1eval, every record fit's read. In-domain: MuSiQue <- WQ-hotpotqa, HotpotQA <- WQ-musique, 2Wiki <- both; held-out the others.

**Call: ADOPT** (ADOPT when the 15 deciding cells are read and none is BELOW).

## The deciding cells (B1's settings: b1v against b1w)

| model | setting | regime | record | D83 | D83 - record (95%) | call | seed 0 | fits below |
|---|---|---|---:|---:|---|---|---|---:|
| zg6r (GNN) | musique | in-domain | 69.0 | 68.5 | -0.5 (-1.2, +0.2) | LEVEL | -0.5 (-1.5, +0.5) | 3/3 |
| zg6r (GNN) | hotpotqa | in-domain | 94.3 | 94.0 | -0.3 (-0.7, +0.1) | LEVEL | -0.5 (-1.1, +0.1) | 3/3 |
| zg6r (GNN) | 2wiki | in-domain | 94.9 | 95.0 | +0.1 (-0.2, +0.3) | LEVEL | -0.1 (-0.4, +0.1) | 4/6 |
| zg6r (GNN) | musique | held-out | 60.8 | 61.6 | +0.8 (+0.2, +1.6) | ABOVE | +0.9 (-0.1, +2.1) | 0/3 |
| zg6r (GNN) | hotpotqa | held-out | 92.9 | 93.0 | +0.1 (-0.4, +0.6) | LEVEL | -0.4 (-1.1, +0.2) | 1/3 |
| GNX (GNN, same inputs) | musique | in-domain | 69.1 | 68.5 | -0.5 (-1.3, +0.2) | LEVEL | -1.2 (-2.2, -0.2) | 2/3 |
| GNX (GNN, same inputs) | hotpotqa | in-domain | 94.1 | 94.3 | +0.3 (-0.1, +0.7) | LEVEL | -0.1 (-0.6, +0.5) | 1/3 |
| GNX (GNN, same inputs) | 2wiki | in-domain | 95.0 | 95.1 | +0.1 (-0.1, +0.4) | LEVEL | +0.2 (-0.1, +0.6) | 2/6 |
| GNX (GNN, same inputs) | musique | held-out | 63.1 | 64.1 | +1.0 (+0.2, +1.8) | ABOVE | +1.4 (+0.2, +2.6) | 0/3 |
| GNX (GNN, same inputs) | hotpotqa | held-out | 93.2 | 92.9 | -0.3 (-0.8, +0.2) | LEVEL | -0.4 (-1.0, +0.3) | 3/3 |
| zrckpn (MLP) | musique | in-domain | 67.4 | 67.9 | +0.4 (-0.2, +1.1) | LEVEL | +0.6 (-0.4, +1.6) | 0/3 |
| zrckpn (MLP) | hotpotqa | in-domain | 93.4 | 93.6 | +0.2 (-0.1, +0.6) | LEVEL | +0.2 (-0.2, +0.8) | 0/3 |
| zrckpn (MLP) | 2wiki | in-domain | 94.1 | 94.4 | +0.3 (+0.0, +0.6) | ABOVE | +0.1 (-0.2, +0.5) | 1/6 |
| zrckpn (MLP) | musique | held-out | 61.8 | 64.5 | +2.7 (+2.0, +3.4) | ABOVE | +4.0 (+3.0, +5.1) | 0/3 |
| zrckpn (MLP) | hotpotqa | held-out | 92.2 | 92.2 | -0.0 (-0.5, +0.4) | LEVEL | -0.1 (-0.9, +0.7) | 2/3 |

## FC@5 and hit@1 on the deciding cells (beside)

| model | setting | regime | FC@5 record | FC@5 D83 | FC@5 diff (95%) | hit@1 record | hit@1 D83 | hit@1 diff (95%) |
|---|---|---|---:|---:|---|---:|---:|---|
| zg6r (GNN) | musique | in-domain | 39.6 | 38.7 | -0.9 (-2.2, +0.4) | 80.4 | 82.0 | +1.6 (+0.4, +2.8) |
| zg6r (GNN) | hotpotqa | in-domain | 89.0 | 88.6 | -0.4 (-1.2, +0.3) | 91.2 | 90.6 | -0.6 (-1.4, +0.1) |
| zg6r (GNN) | 2wiki | in-domain | 87.2 | 87.5 | +0.3 (-0.2, +0.8) | 99.3 | 99.5 | +0.1 (-0.1, +0.4) |
| zg6r (GNN) | musique | held-out | 29.3 | 30.3 | +1.0 (-0.3, +2.3) | 74.7 | 76.3 | +1.6 (+0.4, +2.8) |
| zg6r (GNN) | hotpotqa | held-out | 86.5 | 86.8 | +0.3 (-0.8, +1.2) | 87.3 | 87.6 | +0.3 (-0.5, +1.0) |
| GNX (GNN, same inputs) | musique | in-domain | 40.0 | 38.6 | -1.4 (-2.8, -0.1) | 80.5 | 81.8 | +1.4 (+0.2, +2.5) |
| GNX (GNN, same inputs) | hotpotqa | in-domain | 88.6 | 89.3 | +0.7 (-0.1, +1.5) | 91.1 | 90.8 | -0.3 (-0.9, +0.3) |
| GNX (GNN, same inputs) | 2wiki | in-domain | 87.5 | 87.6 | +0.1 (-0.4, +0.7) | 99.4 | 99.4 | +0.1 (-0.1, +0.2) |
| GNX (GNN, same inputs) | musique | held-out | 31.9 | 32.6 | +0.7 (-0.5, +1.9) | 76.2 | 76.4 | +0.2 (-1.0, +1.3) |
| GNX (GNN, same inputs) | hotpotqa | held-out | 87.1 | 86.9 | -0.2 (-1.1, +0.7) | 87.1 | 86.7 | -0.4 (-1.1, +0.2) |
| zrckpn (MLP) | musique | in-domain | 37.7 | 37.8 | +0.1 (-1.1, +1.4) | 79.3 | 80.5 | +1.2 (+0.2, +2.3) |
| zrckpn (MLP) | hotpotqa | in-domain | 87.4 | 87.8 | +0.5 (-0.2, +1.1) | 90.5 | 90.4 | -0.1 (-0.6, +0.4) |
| zrckpn (MLP) | 2wiki | in-domain | 84.9 | 85.5 | +0.5 (-0.2, +1.3) | 98.9 | 98.7 | -0.3 (-0.6, +0.1) |
| zrckpn (MLP) | musique | held-out | 30.3 | 33.6 | +3.3 (+2.2, +4.5) | 76.8 | 76.2 | -0.6 (-1.6, +0.5) |
| zrckpn (MLP) | hotpotqa | held-out | 85.1 | 85.2 | +0.1 (-0.7, +1.0) | 90.4 | 90.2 | -0.2 (-0.9, +0.5) |

## By hops (MuSiQue) and by kind (2Wiki, HotpotQA): D83 - record, R@5 (beside, descriptive)

| model | setting | regime | group | questions | record | D83 | D83 - record (95%) |
|---|---|---|---|---:|---:|---:|---|
| zg6r (GNN) | musique | in-domain | 2 hops | 518 | 79.0 | 78.9 | -0.1 (-1.3, +0.9) |
| zg6r (GNN) | musique | in-domain | 3 hops | 316 | 64.2 | 63.6 | -0.6 (-1.7, +0.6) |
| zg6r (GNN) | musique | in-domain | 4 hops | 166 | 46.6 | 45.1 | -1.6 (-2.9, -0.3) |
| zg6r (GNN) | hotpotqa | in-domain | bridge | 811 | 93.0 | 92.6 | -0.4 (-0.9, +0.2) |
| zg6r (GNN) | hotpotqa | in-domain | comparison | 189 | 99.9 | 99.9 | +0.0 (+0.0, +0.0) |
| zg6r (GNN) | 2wiki | in-domain | compositional | 413 | 94.7 | 94.9 | +0.2 (-0.2, +0.6) |
| zg6r (GNN) | 2wiki | in-domain | inference | 108 | 93.8 | 93.4 | -0.5 (-1.5, +0.4) |
| zg6r (GNN) | 2wiki | in-domain | comparison | 244 | 100.0 | 100.0 | -0.0 (-0.1, +0.0) |
| zg6r (GNN) | 2wiki | in-domain | bridge_comparison | 235 | 90.5 | 90.7 | +0.2 (-0.2, +0.5) |
| zg6r (GNN) | musique | held-out | 2 hops | 518 | 72.0 | 73.2 | +1.2 (+0.1, +2.4) |
| zg6r (GNN) | musique | held-out | 3 hops | 316 | 54.0 | 54.4 | +0.4 (-0.8, +1.5) |
| zg6r (GNN) | musique | held-out | 4 hops | 166 | 38.7 | 39.2 | +0.5 (-0.5, +1.5) |
| zg6r (GNN) | hotpotqa | held-out | bridge | 811 | 91.2 | 91.4 | +0.1 (-0.5, +0.7) |
| zg6r (GNN) | hotpotqa | held-out | comparison | 189 | 100.0 | 100.0 | +0.0 (+0.0, +0.0) |
| GNX (GNN, same inputs) | musique | in-domain | 2 hops | 518 | 79.2 | 78.9 | -0.3 (-1.4, +0.9) |
| GNX (GNN, same inputs) | musique | in-domain | 3 hops | 316 | 64.3 | 63.6 | -0.8 (-1.9, +0.4) |
| GNX (GNN, same inputs) | musique | in-domain | 4 hops | 166 | 46.6 | 45.8 | -0.8 (-1.9, +0.3) |
| GNX (GNN, same inputs) | hotpotqa | in-domain | bridge | 811 | 92.7 | 93.1 | +0.4 (-0.1, +0.9) |
| GNX (GNN, same inputs) | hotpotqa | in-domain | comparison | 189 | 100.0 | 99.9 | -0.1 (-0.3, +0.0) |
| GNX (GNN, same inputs) | 2wiki | in-domain | compositional | 413 | 95.0 | 95.3 | +0.3 (-0.2, +0.8) |
| GNX (GNN, same inputs) | 2wiki | in-domain | inference | 108 | 93.2 | 94.1 | +0.9 (+0.0, +1.9) |
| GNX (GNN, same inputs) | 2wiki | in-domain | comparison | 244 | 100.0 | 99.9 | -0.0 (-0.1, +0.0) |
| GNX (GNN, same inputs) | 2wiki | in-domain | bridge_comparison | 235 | 90.6 | 90.3 | -0.3 (-0.7, +0.1) |
| GNX (GNN, same inputs) | musique | held-out | 2 hops | 518 | 74.5 | 75.6 | +1.2 (+0.0, +2.3) |
| GNX (GNN, same inputs) | musique | held-out | 3 hops | 316 | 56.3 | 57.5 | +1.2 (+0.0, +2.4) |
| GNX (GNN, same inputs) | musique | held-out | 4 hops | 166 | 40.3 | 40.4 | +0.1 (-1.2, +1.4) |
| GNX (GNN, same inputs) | hotpotqa | held-out | bridge | 811 | 91.7 | 91.3 | -0.4 (-1.0, +0.2) |
| GNX (GNN, same inputs) | hotpotqa | held-out | comparison | 189 | 99.9 | 99.9 | +0.0 (+0.0, +0.0) |
| zrckpn (MLP) | musique | in-domain | 2 hops | 518 | 77.9 | 78.8 | +0.8 (-0.3, +1.9) |
| zrckpn (MLP) | musique | in-domain | 3 hops | 316 | 62.8 | 62.6 | -0.1 (-1.4, +1.0) |
| zrckpn (MLP) | musique | in-domain | 4 hops | 166 | 43.5 | 43.8 | +0.3 (-0.8, +1.4) |
| zrckpn (MLP) | hotpotqa | in-domain | bridge | 811 | 91.9 | 92.2 | +0.3 (-0.1, +0.7) |
| zrckpn (MLP) | hotpotqa | in-domain | comparison | 189 | 99.8 | 99.8 | +0.0 (+0.0, +0.0) |
| zrckpn (MLP) | 2wiki | in-domain | compositional | 413 | 94.1 | 94.8 | +0.7 (+0.1, +1.4) |
| zrckpn (MLP) | 2wiki | in-domain | inference | 108 | 92.4 | 93.3 | +0.9 (+0.0, +2.1) |
| zrckpn (MLP) | 2wiki | in-domain | comparison | 244 | 100.0 | 100.0 | +0.0 (+0.0, +0.0) |
| zrckpn (MLP) | 2wiki | in-domain | bridge_comparison | 235 | 88.7 | 88.4 | -0.3 (-0.9, +0.3) |
| zrckpn (MLP) | musique | held-out | 2 hops | 518 | 72.7 | 76.0 | +3.2 (+2.2, +4.3) |
| zrckpn (MLP) | musique | held-out | 3 hops | 316 | 55.9 | 58.8 | +3.0 (+1.9, +4.0) |
| zrckpn (MLP) | musique | held-out | 4 hops | 166 | 39.2 | 39.6 | +0.4 (-0.6, +1.4) |
| zrckpn (MLP) | hotpotqa | held-out | bridge | 811 | 90.4 | 90.4 | -0.0 (-0.6, +0.5) |
| zrckpn (MLP) | hotpotqa | held-out | comparison | 189 | 100.0 | 100.0 | +0.0 (+0.0, +0.0) |

## The gaps between the models, at D83 beside the record's (R@5, paired by question)

| carve | dataset | regime | gap | record | D83 | change (D83 - record) |
|---|---|---|---|---|---|---|
| b1v | musique | in-domain | gnx-zrckpn | +1.7 (+0.9, +2.4) | +0.7 (+0.0, +1.4) | -1.0 (-1.8, -0.1) |
| b1v | musique | in-domain | zg6r-zrckpn | +1.5 (+0.7, +2.4) | +0.6 (-0.2, +1.4) | -0.9 (-1.8, -0.1) |
| b1v | hotpotqa | in-domain | gnx-zrckpn | +0.7 (+0.2, +1.1) | +0.7 (+0.3, +1.2) | +0.1 (-0.5, +0.6) |
| b1v | hotpotqa | in-domain | zg6r-zrckpn | +0.9 (+0.4, +1.4) | +0.3 (-0.1, +0.8) | -0.5 (-1.1, -0.1) |
| b1v | 2wiki | in-domain | gnx-zrckpn | +0.9 (+0.6, +1.2) | +0.7 (+0.5, +1.0) | -0.2 (-0.5, +0.1) |
| b1v | 2wiki | in-domain | zg6r-zrckpn | +0.8 (+0.5, +1.2) | +0.6 (+0.2, +0.9) | -0.3 (-0.6, +0.1) |
| b1v | musique | held-out | gnx-zrckpn | +1.2 (+0.4, +2.1) | -0.5 (-1.2, +0.3) | -1.7 (-2.6, -0.8) |
| b1v | musique | held-out | zg6r-zrckpn | -1.0 (-1.9, -0.1) | -2.9 (-3.8, -1.9) | -1.8 (-2.8, -0.9) |
| b1v | hotpotqa | held-out | gnx-zrckpn | +1.0 (+0.4, +1.6) | +0.7 (+0.1, +1.2) | -0.3 (-0.9, +0.3) |
| b1v | hotpotqa | held-out | zg6r-zrckpn | +0.6 (+0.1, +1.2) | +0.8 (+0.2, +1.3) | +0.1 (-0.5, +0.8) |
| s1eval | squad | in-domain | gnx-zrckpn | +0.1 (-0.0, +0.2) | -0.0 (-0.1, +0.1) | -0.1 (-0.2, +0.0) |
| s1eval | squad | in-domain | zg6r-zrckpn | +0.1 (-0.0, +0.2) | -0.0 (-0.2, +0.1) | -0.1 (-0.3, -0.0) |
| s1eval | hotpotqa | in-domain | gnx-zrckpn | +1.2 (+1.0, +1.5) | +1.3 (+1.1, +1.5) | +0.1 (-0.2, +0.3) |
| s1eval | hotpotqa | in-domain | zg6r-zrckpn | +1.1 (+0.9, +1.3) | +1.2 (+1.0, +1.5) | +0.1 (-0.1, +0.4) |
| s1eval | hotpotqa | held-out | gnx-zrckpn | -1.6 (-1.9, -1.3) | -1.0 (-1.3, -0.7) | +0.6 (+0.3, +1.0) |
| s1eval | hotpotqa | held-out | zg6r-zrckpn | -2.4 (-2.8, -2.1) | -1.8 (-2.2, -1.5) | +0.6 (+0.3, +0.9) |
| s1eval | 2wiki | in-domain | gnx-zrckpn | +1.2 (+1.0, +1.3) | +1.0 (+0.9, +1.1) | -0.2 (-0.3, -0.1) |
| s1eval | 2wiki | in-domain | zg6r-zrckpn | +1.3 (+1.1, +1.4) | +0.9 (+0.8, +1.0) | -0.4 (-0.5, -0.2) |
| s1eval | musique | in-domain | gnx-zrckpn | +1.5 (+1.0, +2.0) | +0.9 (+0.4, +1.4) | -0.6 (-1.1, -0.1) |
| s1eval | musique | in-domain | zg6r-zrckpn | +1.5 (+1.0, +2.0) | +0.9 (+0.4, +1.5) | -0.6 (-1.1, -0.1) |
| s1eval | musique | held-out | gnx-zrckpn | -0.4 (-0.8, +0.1) | -0.9 (-1.4, -0.4) | -0.5 (-1.0, -0.0) |
| s1eval | musique | held-out | zg6r-zrckpn | -1.8 (-2.3, -1.3) | -3.4 (-3.9, -2.8) | -1.6 (-2.1, -1.0) |
| s1eval | metaqa | held-out | gnx-zrckpn | -3.6 (-4.1, -3.1) | -2.3 (-2.8, -1.8) | +1.3 (+0.9, +1.7) |
| s1eval | metaqa | held-out | zg6r-zrckpn | -5.0 (-5.5, -4.4) | -2.7 (-3.2, -2.3) | +2.2 (+1.8, +2.6) |
| s1eval | webqsp | in-domain | gnx-zrckpn | +0.8 (-0.4, +2.1) | +0.5 (-0.7, +1.6) | -0.3 (-1.0, +0.2) |
| s1eval | webqsp | in-domain | zg6r-zrckpn | +0.7 (-0.4, +1.9) | +0.3 (-1.0, +1.5) | -0.4 (-1.1, +0.2) |

## The six s1eval carves: D83 against every record seed filed (beside, deciding nothing)

| model | dataset | regime | questions | record | D83 | D83 - record (95%) | FC@5 diff | hit@1 diff |
|---|---|---|---:|---:|---:|---|---|---|
| zg6r (GNN) | squad | in-domain | 11873 | 91.4 | 91.4 | -0.0 (-0.1, +0.1) | -0.0 (-0.1, +0.1) | +0.0 (-0.1, +0.2) |
| zg6r (GNN) | hotpotqa | in-domain | 7405 | 85.8 | 85.8 | -0.0 (-0.2, +0.1) | -0.1 (-0.4, +0.3) | -0.5 (-0.8, -0.2) |
| zg6r (GNN) | hotpotqa | held-out | 7405 | 79.3 | 79.7 | +0.4 (+0.1, +0.6) | +0.4 (-0.1, +0.8) | +0.8 (+0.4, +1.1) |
| zg6r (GNN) | 2wiki | in-domain | 12576 | 92.7 | 92.6 | -0.1 (-0.2, -0.0) | -0.3 (-0.5, -0.1) | +0.3 (+0.2, +0.4) |
| zg6r (GNN) | musique | in-domain | 2417 | 59.5 | 59.4 | -0.0 (-0.5, +0.3) | -1.1 (-1.8, -0.4) | -0.4 (-1.1, +0.2) |
| zg6r (GNN) | musique | held-out | 2417 | 50.9 | 51.5 | +0.7 (+0.2, +1.1) | +1.0 (+0.3, +1.5) | +1.9 (+1.2, +2.7) |
| zg6r (GNN) | metaqa | held-out | 9785 | 36.0 | 39.9 | +3.8 (+3.5, +4.1) | +3.1 (+2.8, +3.4) | +1.7 (+1.4, +1.9) |
| zg6r (GNN) | webqsp | in-domain | 1503 | 65.8 | 65.2 | -0.6 (-1.1, -0.1) | -0.4 (-0.9, +0.2) | -0.6 (-1.4, +0.2) |
| GNX (GNN, same inputs) | squad | in-domain | 11873 | 91.4 | 91.4 | +0.0 (-0.1, +0.1) | +0.0 (-0.1, +0.1) | +0.1 (-0.1, +0.3) |
| GNX (GNN, same inputs) | hotpotqa | in-domain | 7405 | 86.0 | 85.9 | -0.1 (-0.3, +0.1) | -0.1 (-0.5, +0.3) | -0.3 (-0.7, -0.0) |
| GNX (GNN, same inputs) | hotpotqa | held-out | 7405 | 80.1 | 80.5 | +0.4 (+0.2, +0.7) | +0.7 (+0.2, +1.1) | +0.6 (+0.3, +1.0) |
| GNX (GNN, same inputs) | 2wiki | in-domain | 12576 | 92.6 | 92.7 | +0.0 (-0.1, +0.1) | +0.0 (-0.2, +0.3) | +0.0 (-0.1, +0.1) |
| GNX (GNN, same inputs) | musique | in-domain | 2417 | 59.5 | 59.4 | -0.1 (-0.5, +0.3) | -0.4 (-1.1, +0.4) | +0.0 (-0.6, +0.6) |
| GNX (GNN, same inputs) | musique | held-out | 2417 | 52.3 | 54.0 | +1.7 (+1.2, +2.2) | +1.8 (+1.1, +2.5) | +0.1 (-0.6, +0.8) |
| GNX (GNN, same inputs) | metaqa | held-out | 9785 | 37.4 | 40.3 | +2.9 (+2.6, +3.2) | +2.2 (+1.9, +2.5) | +2.8 (+2.4, +3.1) |
| GNX (GNN, same inputs) | webqsp | in-domain | 1503 | 66.0 | 65.5 | -0.5 (-1.0, -0.1) | -0.6 (-1.1, -0.0) | -1.3 (-2.1, -0.5) |
| zrckpn (MLP) | squad | in-domain | 11873 | 91.3 | 91.5 | +0.1 (+0.0, +0.2) | +0.1 (+0.0, +0.2) | -0.1 (-0.2, +0.1) |
| zrckpn (MLP) | hotpotqa | in-domain | 7405 | 84.7 | 84.6 | -0.2 (-0.3, +0.0) | -0.2 (-0.5, +0.1) | +0.5 (+0.3, +0.8) |
| zrckpn (MLP) | hotpotqa | held-out | 7405 | 81.7 | 81.5 | -0.2 (-0.4, +0.0) | -0.5 (-1.0, -0.1) | +0.3 (-0.0, +0.6) |
| zrckpn (MLP) | 2wiki | in-domain | 12576 | 91.5 | 91.7 | +0.2 (+0.1, +0.3) | +0.7 (+0.5, +1.0) | -0.0 (-0.1, +0.1) |
| zrckpn (MLP) | musique | in-domain | 2417 | 58.0 | 58.5 | +0.5 (+0.1, +0.9) | +0.5 (-0.1, +1.2) | +0.6 (-0.0, +1.2) |
| zrckpn (MLP) | musique | held-out | 2417 | 52.7 | 54.9 | +2.2 (+1.8, +2.7) | +2.6 (+2.0, +3.2) | +1.2 (+0.5, +1.9) |
| zrckpn (MLP) | metaqa | held-out | 9785 | 41.0 | 42.6 | +1.6 (+1.2, +1.9) | +1.3 (+1.0, +1.7) | +0.7 (+0.4, +1.1) |
| zrckpn (MLP) | webqsp | in-domain | 1503 | 65.1 | 65.0 | -0.2 (-0.6, +0.3) | -0.3 (-0.8, +0.3) | +0.8 (+0.2, +1.5) |

## Links per label (unique pairs per question), gold-touching (one end gold) and gold-gold (both ends gold)

**b1v musique** (1000 questions)

| store | label | links | gold-touching | gold-gold | questions with a gold-gold link |
|---|---|---:|---:|---:|---:|
| D83 | S | 0 | 0 | 0 | 0 |
| D83 | R.mention | 13353884 | 26512 | 479 | 400 |
| D83 | R.name_comention | 18651908 | 54727 | 1008 | 748 |
| D83 | R.title_comention | 1206545 | 3964 | 90 | 87 |
| D83 | K.knn | 5198733 | 12495 | 355 | 349 |
| record | slot | 13351661 | 26512 | 479 | 400 |
| record | NER.spacy | 16108552 | 49453 | 956 | 737 |
| record | K.knn | 5203077 | 12519 | 355 | 349 |

**b1v hotpotqa** (1000 questions)

| store | label | links | gold-touching | gold-gold | questions with a gold-gold link |
|---|---|---:|---:|---:|---:|
| D83 | S | 0 | 0 | 0 | 0 |
| D83 | R.mention | 14721 | 2882 | 575 | 575 |
| D83 | R.name_comention | 106970 | 10998 | 581 | 581 |
| D83 | R.title_comention | 2218 | 504 | 109 | 109 |
| D83 | K.knn | 62369 | 7081 | 523 | 523 |
| record | slot | 14721 | 2882 | 575 | 575 |
| record | NER.spacy | 87502 | 8917 | 492 | 492 |
| record | K.knn | 62369 | 7081 | 523 | 523 |

**b1v 2wiki** (1000 questions)

| store | label | links | gold-touching | gold-gold | questions with a gold-gold link |
|---|---|---:|---:|---:|---:|
| D83 | S | 0 | 0 | 0 | 0 |
| D83 | R.mention | 16851 | 1880 | 808 | 643 |
| D83 | R.name_comention | 115593 | 9510 | 767 | 633 |
| D83 | R.title_comention | 2291 | 301 | 124 | 118 |
| D83 | K.knn | 74448 | 6400 | 579 | 532 |
| record | slot | 16851 | 1880 | 808 | 643 |
| record | NER.spacy | 107634 | 9093 | 734 | 613 |
| record | K.knn | 74448 | 6400 | 579 | 532 |

**s1eval squad** (11873 questions)

| store | label | links | gold-touching | gold-gold | questions with a gold-gold link |
|---|---|---:|---:|---:|---:|
| D83 | S | 0 | 0 | 0 | 0 |
| D83 | R.mention | 3051424 | 171563 | 0 | 0 |
| D83 | R.name_comention | 668679 | 45581 | 0 | 0 |
| D83 | R.title_comention | 11846 | 733 | 0 | 0 |
| D83 | K.knn | 490470 | 34983 | 0 | 0 |
| record | slot | 3051424 | 171563 | 0 | 0 |
| record | NER.spacy | 632503 | 43170 | 0 | 0 |
| record | K.knn | 490470 | 34983 | 0 | 0 |

**s1eval hotpotqa** (7405 questions)

| store | label | links | gold-touching | gold-gold | questions with a gold-gold link |
|---|---|---:|---:|---:|---:|
| D83 | S | 0 | 0 | 0 | 0 |
| D83 | R.mention | 9121121 | 287237 | 4240 | 4240 |
| D83 | R.name_comention | 5183524 | 148122 | 1905 | 1905 |
| D83 | R.title_comention | 286959 | 14647 | 816 | 816 |
| D83 | K.knn | 3606137 | 76236 | 1466 | 1466 |
| record | slot | 9121020 | 287227 | 4240 | 4240 |
| record | NER.spacy | 5247505 | 156741 | 1632 | 1632 |
| record | K.knn | 3606340 | 76257 | 1466 | 1466 |

**s1eval 2wiki** (12576 questions)

| store | label | links | gold-touching | gold-gold | questions with a gold-gold link |
|---|---|---:|---:|---:|---:|
| D83 | S | 0 | 0 | 0 | 0 |
| D83 | R.mention | 25975007 | 628965 | 10716 | 8474 |
| D83 | R.name_comention | 15743744 | 516706 | 5830 | 5424 |
| D83 | R.title_comention | 599902 | 22064 | 1019 | 1002 |
| D83 | K.knn | 6985181 | 134294 | 2800 | 2757 |
| record | slot | 25974694 | 628948 | 10716 | 8474 |
| record | NER.spacy | 14965472 | 543752 | 5287 | 4900 |
| record | K.knn | 6985987 | 134309 | 2800 | 2757 |

**s1eval musique** (2417 questions)

| store | label | links | gold-touching | gold-gold | questions with a gold-gold link |
|---|---|---:|---:|---:|---:|
| D83 | S | 0 | 0 | 0 | 0 |
| D83 | R.mention | 167022173 | 276058 | 1026 | 892 |
| D83 | R.name_comention | 44413984 | 110571 | 1267 | 1223 |
| D83 | R.title_comention | 2259862 | 4714 | 111 | 111 |
| D83 | K.knn | 15213023 | 28745 | 430 | 425 |
| record | slot | 166513397 | 273524 | 1020 | 889 |
| record | NER.spacy | 42832932 | 109046 | 1188 | 1155 |
| record | K.knn | 15376146 | 29226 | 431 | 426 |

**s1eval metaqa** (9785 questions)

| store | label | links | gold-touching | gold-gold | questions with a gold-gold link |
|---|---|---:|---:|---:|---:|
| D83 | S | 119272476 | 3729324 | 587 | 423 |
| D83 | R.mention | 0 | 0 | 0 | 0 |
| D83 | R.name_comention | 4190038 | 7479 | 343 | 131 |
| D83 | R.title_comention | 0 | 0 | 0 | 0 |
| D83 | K.knn | 31261529 | 123429 | 11859 | 1496 |
| record | slot | 119272476 | 3729324 | 587 | 423 |
| record | NER.spacy | 3007453 | 14146 | 123 | 39 |
| record | K.knn | 31261529 | 123429 | 11859 | 1496 |

**s1eval webqsp** (1503 questions)

| store | label | links | gold-touching | gold-gold | questions with a gold-gold link |
|---|---|---:|---:|---:|---:|
| D83 | S | 14697542 | 466823 | 1426 | 228 |
| D83 | R.mention | 0 | 0 | 0 | 0 |
| D83 | R.name_comention | 979487 | 7402 | 1342 | 60 |
| D83 | R.title_comention | 1177 | 5 | 0 | 0 |
| D83 | K.knn | 6663039 | 22650 | 2040 | 284 |
| record | slot | 14697542 | 466823 | 1426 | 228 |
| record | NER.spacy | 3356815 | 9517 | 202 | 21 |
| record | K.knn | 6663451 | 22686 | 2052 | 290 |


Missing reads: 0; G6 problems: 0.
