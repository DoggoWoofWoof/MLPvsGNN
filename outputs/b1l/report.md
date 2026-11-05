# B1L report (declared in docs/B1L_MODELS_OF_RECORD_ON_THE_LLM_GRAPH.md)

R@5 x100 of the p@swa candidate on HippoRAG 2's settings (1,000 questions each), no new fit. The two models of record, zg6r and zrckpn, were fitted without LLM links (on W2_S pools) and are read on three link sets: b1t (the text links only), b1w (the rule's W2_S links, W2_S-typed) and b1o (the text links plus the P pairs of the setting's LLM-built reference graph: passages sharing a phrase that occurs in 2 to 25 passages, each pair once each way; the 1/df weights do not enter, the structural family being unweighted by contract, and their distribution is recorded beside). In-domain = the mean over the fits that trained on the dataset; 95% bootstrap intervals over the questions (2,000 resamples, seed 0).

**What the calls say.** b1o's entries are NULL-heavy (most have no W2_S type) while b1w's are W2_S-typed and the fits were fitted on W2_S types, so b1o - b1t is a lower bound on what these models can take from an LLM-built graph, and b1w - b1o mixes the pair set with the typing. The calls compare how the models of record, fitted without LLM links, read on the rule's links and on the LLM-built pair set; they are not a statement that the rule's links beat the LLM's.

## In-domain

| setting | model | b1t | b1w | b1o | b1o - b1t | b1w - b1t | b1w - b1o | calls (o-t / w-t / w-o) | share of the LLM step the rule's step recovers |
|---|---|---:|---:|---:|---|---|---|---|---|
| musique | zg6r | 66.0 (64.3-67.7) | 69.0 (67.2-70.7) | 67.4 (65.8-69.3) | +1.5 (+0.4, +2.6) | +3.0 (+1.8, +4.0) | +1.5 (+0.4, +2.5) | UP / UP / GREATER | 2.03 (1.20-6.02) |
| musique | zrckpn | 64.5 (62.8-66.1) | 67.4 (65.7-69.2) | 67.3 (65.6-69.2) | +2.8 (+1.7, +3.9) | +2.9 (+1.9, +4.0) | +0.1 (-0.9, +1.1) | UP / UP / SAME | 1.03 (0.71-1.53) |
| 2wiki | zg6r | 78.1 (76.7-79.5) | 94.9 (94.1-95.7) | 89.8 (88.7-90.8) | +11.7 (+10.4, +13.1) | +16.8 (+15.5, +18.2) | +5.1 (+4.0, +6.3) | UP / UP / GREATER | 1.44 (1.32-1.57) |
| 2wiki | zrckpn | 77.0 (75.6-78.4) | 94.1 (93.2-94.9) | 90.8 (89.8-91.8) | +13.8 (+12.5, +15.1) | +17.1 (+15.8, +18.4) | +3.3 (+2.3, +4.3) | UP / UP / GREATER | 1.24 (1.16-1.33) |

## Held-out (fits that did not train on the dataset)

| setting | model | b1t | b1w | b1o | b1o - b1t | b1w - b1t | b1w - b1o | calls (o-t / w-t / w-o) | share of the LLM step the rule's step recovers |
|---|---|---:|---:|---:|---|---|---|---|---|
| musique | zg6r | 62.1 (60.4-63.7) | 60.8 (59.1-62.5) | 62.8 (61.1-64.5) | +0.8 (-0.5, +2.0) | -1.3 (-2.6, +0.1) | -2.0 (-3.5, -0.6) | FLAT / FLAT / LESS | n/a: b1o - b1t not above 0 |
| musique | zrckpn | 60.0 (58.4-61.7) | 61.8 (60.1-63.6) | 64.0 (62.3-65.8) | +4.0 (+2.7, +5.2) | +1.8 (+0.5, +3.1) | -2.1 (-3.5, -0.8) | UP / UP / LESS | 0.46 (0.17-0.75) |

## The call per setting (in-domain b1w - b1o, both models)

| setting | call | zg6r | zrckpn |
|---|---|---|---|
| musique | NOT_LESS | GREATER | SAME |
| 2wiki | GREATER | GREATER | GREATER |

## zg6r - zrckpn on each carve, paired by question

| setting | read | carve | zg6r - zrckpn (CI) | label |
|---|---|---|---|---|
| musique | in-domain | b1t | +1.5 (+0.6, +2.4) | GNN_ABOVE |
| musique | in-domain | b1w | +1.5 (+0.7, +2.4) | GNN_ABOVE |
| musique | in-domain | b1o | +0.1 (-0.7, +0.9) | LEVEL |
| musique | zero-shot | b1t | +2.0 (+1.1, +2.9) | GNN_ABOVE |
| musique | zero-shot | b1w | -1.1 (-1.9, -0.1) | MLP_ABOVE |
| musique | zero-shot | b1o | -1.2 (-2.0, -0.3) | MLP_ABOVE |
| 2wiki | in-domain | b1t | +1.1 (+0.7, +1.6) | GNN_ABOVE |
| 2wiki | in-domain | b1w | +0.8 (+0.5, +1.2) | GNN_ABOVE |
| 2wiki | in-domain | b1o | -1.0 (-1.6, -0.4) | MLP_ABOVE |

## The link sets, by entry type

| setting | reference | P pairs | entries | entries that are W2_S entries | entries with the null score type | W2_S pairs in the setting | W2_S entries that are P entries |
|---|---|---:|---:|---:|---:|---:|---:|
| musique | H4o | 98211 | 196422 | 25014 (12.7%) | 175802 (89.5%) | 41417 | 25014 |
| 2wiki | SR | 27364 | 54728 | 3524 (6.4%) | 51204 (93.6%) | 2030 | 3524 |

Best published rows: musique SAG2 80.68 / PropRAG 78.3; 2wiki EviReform 98.45 / G-reasoner 98.2

Check: the b1t and b1w rows recomputed from B1P's and B1M's read files equal B1M's filed rows (60 checked, 0 differ). Missing reads: 0.
