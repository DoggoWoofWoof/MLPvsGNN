# B1H report (declared in docs/B1H_MODELS_OF_RECORD_ON_HYPERLINKS.md)

R@5 x100 of the p@swa candidate on HippoRAG 2's settings (1,000 questions each), no LLM. The ladder b1t (text links only) -> b1w (U2b's W2_S rule cut to the setting) -> b1 (B1a's corpus hyperlinks; musique: title mentions) for the two models of record, zg6r and zrckpn, trained on W2_S pools and never on a hyperlink graph (a shift of graph family on b1). In-domain = the mean over the fits that trained on the dataset (three seeds; six on 2wiki). 95% bootstrap intervals over the questions (2,000 resamples, seed 0).

## The ladder, in-domain

| setting | model | b1t | b1w | b1 | b1w - b1t | b1 - b1w | b1 - b1t | share of the gain at b1w | b1 calls (any / peer) | b1 gap (any / peer) |
|---|---|---:|---:|---|---|---|---|---:|---|---|
| 2wiki | zg6r | 78.1 (76.7-79.5) | 94.9 (94.1-95.7) | 95.4 (94.6-96.1) | +16.8 (+15.5, +18.2) | +0.4 (-0.3, +1.1) | +17.3 (+16.0, +18.6) | 0.974 | BELOW / BELOW | -3.1 / -2.8 |
| 2wiki | zrckpn | 77.0 (75.6-78.4) | 94.1 (93.2-94.9) | 94.4 (93.5-95.2) | +17.1 (+15.8, +18.4) | +0.3 (-0.4, +0.9) | +17.4 (+16.1, +18.7) | 0.984 | BELOW / BELOW | -4.1 / -3.8 |
| hotpotqa | zg6r | 91.1 (90.0-92.3) | 94.3 (93.3-95.3) | 96.5 (95.7-97.3) | +3.1 (+2.4, +4.0) | +2.2 (+1.5, +2.9) | +5.3 (+4.4, +6.2) | 0.589 | BELOW / BELOW | -2.3 / -0.9 |
| hotpotqa | zrckpn | 90.4 (89.2-91.6) | 93.4 (92.4-94.5) | 95.6 (94.8-96.5) | +3.0 (+2.2, +3.9) | +2.2 (+1.5, +2.9) | +5.2 (+4.3, +6.2) | 0.580 | BELOW / BELOW | -3.1 / -1.8 |
| musique | zg6r | 66.0 (64.3-67.7) | 69.0 (67.2-70.7) | 68.6 (66.9-70.4) | +3.0 (+1.8, +4.0) | -0.4 (-0.9, +0.2) | +2.6 (+1.6, +3.7) | 1.135 | BELOW / BELOW | -12.1 / -9.7 |
| musique | zrckpn | 64.5 (62.8-66.1) | 67.4 (65.7-69.2) | 67.3 (65.6-69.0) | +2.9 (+1.9, +4.0) | -0.1 (-0.6, +0.4) | +2.9 (+1.8, +4.0) | 1.032 | BELOW / BELOW | -13.3 / -11.0 |

Best published rows: 2wiki EviReform 98.45 / G-reasoner 98.2; hotpotqa BridgeRAG 98.75 / PropRAG 97.4; musique SAG2 80.68 / PropRAG 78.3

## The held-out read (fits that did not train on the dataset)

| setting | model | b1t | b1w | b1 | b1w - b1t | b1 - b1w | b1 - b1t |
|---|---|---:|---:|---|---|---|---|
| hotpotqa | zg6r | 89.4 (88.2-90.6) | 92.9 (91.9-94.0) | 94.7 (93.8-95.6) | +3.5 (+2.6, +4.4) | +1.8 (+1.1, +2.5) | +5.3 (+4.3, +6.3) |
| hotpotqa | zrckpn | 88.6 (87.4-89.9) | 92.2 (91.2-93.4) | 93.9 (92.9-94.9) | +3.6 (+2.8, +4.5) | +1.6 (+0.9, +2.3) | +5.3 (+4.4, +6.2) |
| musique | zg6r | 62.1 (60.4-63.7) | 60.8 (59.1-62.5) | 59.1 (57.4-60.9) | -1.3 (-2.6, +0.1) | -1.7 (-2.3, -1.0) | -2.9 (-4.3, -1.6) |
| musique | zrckpn | 60.0 (58.4-61.7) | 61.8 (60.1-63.6) | 61.1 (59.3-62.9) | +1.8 (+0.5, +3.1) | -0.8 (-1.3, -0.1) | +1.1 (-0.1, +2.3) |

## zg6r - zrckpn on each carve, paired by question

| setting | read | carve | zg6r - zrckpn (CI) | label |
|---|---|---|---|---|
| 2wiki | in-domain | b1t | +1.1 (+0.7, +1.6) | GNN_ABOVE |
| 2wiki | in-domain | b1w | +0.8 (+0.5, +1.2) | GNN_ABOVE |
| 2wiki | in-domain | b1 | +1.0 (+0.6, +1.4) | GNN_ABOVE |
| hotpotqa | in-domain | b1t | +0.8 (+0.2, +1.4) | GNN_ABOVE |
| hotpotqa | in-domain | b1w | +0.9 (+0.4, +1.4) | GNN_ABOVE |
| hotpotqa | in-domain | b1 | +0.9 (+0.4, +1.3) | GNN_ABOVE |
| hotpotqa | zero-shot | b1t | +0.8 (+0.0, +1.5) | GNN_ABOVE |
| hotpotqa | zero-shot | b1w | +0.6 (+0.1, +1.2) | GNN_ABOVE |
| hotpotqa | zero-shot | b1 | +0.8 (+0.2, +1.4) | GNN_ABOVE |
| musique | in-domain | b1t | +1.5 (+0.6, +2.4) | GNN_ABOVE |
| musique | in-domain | b1w | +1.5 (+0.7, +2.4) | GNN_ABOVE |
| musique | in-domain | b1 | +1.3 (+0.4, +2.1) | GNN_ABOVE |
| musique | zero-shot | b1t | +2.0 (+1.1, +2.9) | GNN_ABOVE |
| musique | zero-shot | b1w | -1.1 (-1.9, -0.1) | MLP_ABOVE |
| musique | zero-shot | b1 | -1.9 (-2.8, -1.1) | MLP_ABOVE |

## HLC family H on the models of record (call: MIXED; J5's hyperlink-trained fits beside: GREATER_EVERYWHERE)

Model lift = R@5(b1) - R@5(b1t); walk lift = A1 on structural + NER + KNN minus A1 on NER + KNN (HLW's walk); difference = model lift - walk lift, paired by question. Deciding cells (in-domain): 2wiki zg6r, 2wiki zrckpn, hotpotqa zg6r, hotpotqa zrckpn.

| setting | model | read | model lift (CI) | walk lift (CI) | difference (CI) | call |
|---|---|---|---|---|---|---|
| 2wiki | zg6r | in-domain | +17.3 (+16.0, +18.6) | -0.1 (-0.9, +0.6) | +17.4 (+15.9, +18.9) | GREATER |
| 2wiki | zrckpn | in-domain | +17.4 (+16.1, +18.7) | -0.1 (-0.9, +0.6) | +17.5 (+16.0, +19.1) | GREATER |
| hotpotqa | zg6r | in-domain | +5.3 (+4.4, +6.2) | +5.1 (+4.1, +6.3) | +0.2 (-1.2, +1.6) | SAME |
| hotpotqa | zrckpn | in-domain | +5.2 (+4.3, +6.2) | +5.1 (+4.1, +6.3) | +0.1 (-1.4, +1.6) | SAME |
| hotpotqa | zg6r | held-out | +5.3 (+4.3, +6.3) | +5.1 (+4.1, +6.3) | +0.1 (-1.3, +1.6) | SAME |
| hotpotqa | zrckpn | held-out | +5.3 (+4.4, +6.2) | +5.1 (+4.1, +6.3) | +0.1 (-1.4, +1.6) | SAME |
| musique | zg6r | in-domain | +2.6 (+1.6, +3.7) | +0.8 (-0.1, +1.8) | +1.8 (+0.4, +3.1) | GREATER |
| musique | zrckpn | in-domain | +2.9 (+1.8, +4.0) | +0.8 (-0.1, +1.8) | +2.0 (+0.6, +3.4) | GREATER |
| musique | zg6r | held-out | -2.9 (-4.3, -1.6) | +0.8 (-0.1, +1.8) | -3.8 (-5.3, -2.2) | LESS |
| musique | zrckpn | held-out | +1.1 (-0.1, +2.3) | +0.8 (-0.1, +1.8) | +0.2 (-1.2, +1.7) | SAME |

## What b1 added, by entry type

| setting | hyperlink pairs | entries | entries that are W2_S entries | entries with the null score type | W2_S pairs in the setting |
|---|---:|---:|---:|---:|---:|
| 2wiki | 1984 | 3968 | 3348 (84.4%) | 620 (15.6%) | 2030 |
| hotpotqa | 4400 | 8800 | 7074 (80.4%) | 1726 (19.6%) | 4315 |
| musique | 42823 | 85646 | 76508 (89.3%) | 9806 (11.5%) | 41417 |

Check: the b1t and b1w rows recomputed from B1P's and B1M's read files equal B1M's filed rows (92 checked, 0 differ). Missing reads: 0.
