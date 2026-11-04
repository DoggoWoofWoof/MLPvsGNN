# B1M report (declared in docs/B1M_MLP_OF_RECORD_ON_B1.md)

R@5 x100 of the p@swa candidate on HippoRAG 2's settings (1,000 questions each), no LLM and no hyperlink. b1w = U2b's W2_S rule cut to the setting; b1t = the setting's text links only. zg6r and zrckp are B1P's reads; zrckpn is B1M's. In-domain = the mean over the fits that trained on the dataset (three seeds; six fits on 2wiki). 95% bootstrap intervals over the questions.

## In-domain

| setting | model | b1t | b1w (CI) | w - t (CI) | b1w FC@5 | b1w hit@1 | calls (any / peer) | gap (any / peer) |
|---|---|---:|---:|---|---:|---:|---|---|
| musique | zg6r | 66.0 | 69.0 (67.2-70.7) | +3.0 (+1.8, +4.0) | 39.6 | 80.4 | BELOW / BELOW | -11.7 / -9.3 |
| musique | zrckpn | 64.5 | 67.4 (65.7-69.2) | +2.9 (+1.9, +4.0) | 37.7 | 79.3 | BELOW / BELOW | -13.2 / -10.9 |
| musique | zrckp | 62.5 | 66.9 (65.2-68.6) | +4.4 (+3.3, +5.6) | 36.8 | 79.8 | BELOW / BELOW | -13.8 / -11.4 |
| 2wiki | zg6r | 78.1 | 94.9 (94.1-95.7) | +16.8 (+15.5, +18.2) | 87.2 | 99.3 | BELOW / BELOW | -3.5 / -3.3 |
| 2wiki | zrckpn | 77.0 | 94.1 (93.2-94.9) | +17.1 (+15.8, +18.4) | 84.9 | 98.9 | BELOW / BELOW | -4.4 / -4.1 |
| 2wiki | zrckp | 75.9 | 94.0 (93.1-94.8) | +18.1 (+16.8, +19.4) | 84.8 | 98.9 | BELOW / BELOW | -4.5 / -4.2 |
| hotpotqa | zg6r | 91.1 | 94.3 (93.3-95.3) | +3.1 (+2.4, +4.0) | 89.0 | 91.2 | BELOW / BELOW | -4.5 / -3.1 |
| hotpotqa | zrckpn | 90.4 | 93.4 (92.4-94.5) | +3.0 (+2.2, +3.9) | 87.4 | 90.5 | BELOW / BELOW | -5.3 / -4.0 |
| hotpotqa | zrckp | 90.0 | 93.4 (92.4-94.5) | +3.4 (+2.5, +4.3) | 87.3 | 90.5 | BELOW / BELOW | -5.3 / -4.0 |

Best published rows: musique SAG2 80.68 / PropRAG 78.3; 2wiki EviReform 98.45 / G-reasoner 98.2; hotpotqa BridgeRAG 98.75 / PropRAG 97.4

## The two models of record: zg6r - zrckpn, paired by question

| setting | read | carve | zg6r | zrckpn | zg6r - zrckpn (CI) | label | per fit pair (same seed and split) |
|---|---|---|---:|---:|---|---|---|
| musique | in-domain | b1w | 69.0 | 67.4 | +1.5 (+0.7, +2.4) | GNN_ABOVE | +1.0, +2.5, +1.1 |
| musique | in-domain | b1t | 66.0 | 64.5 | +1.5 (+0.6, +2.4) | GNN_ABOVE | +1.1, +2.1, +1.2 |
| musique | zero-shot | b1w | 60.8 | 61.8 | -1.1 (-1.9, -0.1) | MLP_ABOVE | +0.6, -1.6, -2.1 |
| musique | zero-shot | b1t | 62.1 | 60.0 | +2.0 (+1.1, +2.9) | GNN_ABOVE | +3.6, +0.7, +1.8 |
| 2wiki | in-domain | b1w | 94.9 | 94.1 | +0.8 (+0.5, +1.2) | GNN_ABOVE | -0.2, +0.2, -0.1, +1.7, +1.6, +1.8 |
| 2wiki | in-domain | b1t | 78.1 | 77.0 | +1.1 (+0.7, +1.6) | GNN_ABOVE | +1.2, +1.8, +0.6, +0.3, +1.6, +1.3 |
| hotpotqa | in-domain | b1w | 94.3 | 93.4 | +0.9 (+0.4, +1.4) | GNN_ABOVE | +0.9, +0.7, +1.1 |
| hotpotqa | in-domain | b1t | 91.1 | 90.4 | +0.8 (+0.2, +1.4) | GNN_ABOVE | +0.6, +0.9, +0.8 |
| hotpotqa | zero-shot | b1w | 92.9 | 92.2 | +0.6 (+0.1, +1.2) | GNN_ABOVE | +0.4, +1.1, +0.4 |
| hotpotqa | zero-shot | b1t | 89.4 | 88.6 | +0.8 (+0.0, +1.5) | GNN_ABOVE | -0.1, +1.7, +0.8 |

## Beside, deciding nothing

| setting | read | carve | zrckpn - zrckp (CI) | zrckpn w - t (CI) |
|---|---|---|---|---|
| musique | in-domain | b1w | +0.5 (-0.1, +1.2) | +2.9 (+1.9, +4.0) |
| musique | in-domain | b1t | +2.0 (+1.3, +2.7) |  |
| musique | zero-shot | b1w | +1.0 (+0.3, +1.6) | +1.8 (+0.5, +3.1) |
| musique | zero-shot | b1t | +1.4 (+0.7, +2.1) |  |
| 2wiki | in-domain | b1w | +0.1 (-0.2, +0.5) | +17.1 (+15.8, +18.4) |
| 2wiki | in-domain | b1t | +1.1 (+0.6, +1.6) |  |
| hotpotqa | in-domain | b1w | -0.0 (-0.4, +0.4) | +3.0 (+2.2, +3.9) |
| hotpotqa | in-domain | b1t | +0.3 (-0.3, +0.9) |  |
| hotpotqa | zero-shot | b1w | +0.1 (-0.4, +0.5) | +3.6 (+2.8, +4.5) |
| hotpotqa | zero-shot | b1t | +0.7 (+0.1, +1.4) |  |

Zero-shot reads (R@5 x100, b1w): musique zg6r 60.8; musique zrckpn 61.8; musique zrckp 60.9; hotpotqa zg6r 92.9; hotpotqa zrckpn 92.2; hotpotqa zrckp 92.2

Training overlap (B1P's counts): musique {'WQ-hotpotqa:musique/s1fit': 0}; 2wiki {'WQ-musique:2wiki/fit': 0, 'WQ-hotpotqa:2wiki/fit': 0}; hotpotqa {'WQ-musique:hotpotqa/fit': 0}

LTM's fits name WQ's training carves: {'scr-ltm-zrckpn-s0': True, 'scr-ltm-zrckpn-s0-hp': True, 'scr-ltm-zrckpn-s1': True, 'scr-ltm-zrckpn-s1-hp': True, 'scr-ltm-zrckpn-s2': True, 'scr-ltm-zrckpn-s2-hp': True}

Missing reads: 0.
