# AP2 report (declared in docs/AP2_RULE_NER_SLOT_ON_B1.md)

R@5 x100 on HippoRAG 2's settings (1,000 questions each), no new fit. The models of record (zg6r, zrckpn) are read on b1t (N + K), b1w (R + spaCy's N + K) and **b1n (R + F0T + K)**: the NER slot holds the rule's names with each passage's own title (AP1b), the other files are b1w's. A read-time shift: the models were fitted on spaCy's NER store. In-domain = the mean over the fits that trained on the setting's partner split. 95% bootstrap intervals over the questions (2,000 resamples, seed 0); words by the interval, no threshold (GREATER / SAME / LESS for b1n - b1w; UP / DOWN / FLAT against b1t).

**Verdict (Part A): REPLACE** (REPLACE when no setting's call is LESS or MIXED).

## In-domain

| setting | model | b1t | b1w | b1n | b1n - b1w | b1n - b1t | b1w - b1t | word (b1n - b1w) |
|---|---|---:|---:|---:|---|---|---|---|
| 2wiki | zg6r | 78.1 (76.7-79.5) | 94.9 (94.1-95.7) | 94.7 (93.9-95.5) | -0.2 (-0.5, -0.0) | +16.6 (+15.3, +17.9) | +16.8 (+15.5, +18.2) | SAME |
| 2wiki | zrckpn | 77.0 (75.6-78.4) | 94.1 (93.2-94.9) | 94.0 (93.2-94.8) | -0.1 (-0.4, +0.2) | +17.0 (+15.7, +18.4) | +17.1 (+15.8, +18.4) | SAME |
| hotpotqa | zg6r | 91.1 (90.0-92.3) | 94.3 (93.3-95.3) | 94.3 (93.3-95.4) | +0.0 (-0.2, +0.3) | +3.2 (+2.4, +4.0) | +3.1 (+2.4, +4.0) | SAME |
| hotpotqa | zrckpn | 90.4 (89.2-91.6) | 93.4 (92.4-94.5) | 93.5 (92.4-94.5) | +0.1 (-0.1, +0.2) | +3.1 (+2.2, +4.0) | +3.0 (+2.2, +3.9) | SAME |
| musique | zg6r | 66.0 (64.3-67.7) | 69.0 (67.2-70.7) | 68.7 (67.0-70.4) | -0.3 (-0.7, +0.2) | +2.7 (+1.6, +3.8) | +3.0 (+1.8, +4.0) | SAME |
| musique | zrckpn | 64.5 (62.8-66.1) | 67.4 (65.7-69.2) | 67.2 (65.5-68.9) | -0.2 (-0.7, +0.2) | +2.7 (+1.6, +3.8) | +2.9 (+1.9, +4.0) | SAME |

## Held-out

| setting | model | b1t | b1w | b1n | b1n - b1w | b1n - b1t | b1w - b1t | word (b1n - b1w) |
|---|---|---:|---:|---:|---|---|---|---|
| hotpotqa | zg6r | 89.4 (88.2-90.6) | 92.9 (91.9-94.0) | 92.9 (91.8-94.0) | -0.0 (-0.3, +0.3) | +3.5 (+2.6, +4.4) | +3.5 (+2.6, +4.4) | SAME |
| hotpotqa | zrckpn | 88.6 (87.4-89.9) | 92.2 (91.2-93.4) | 92.1 (91.0-93.2) | -0.2 (-0.5, +0.1) | +3.5 (+2.6, +4.3) | +3.6 (+2.8, +4.5) | SAME |
| musique | zg6r | 62.1 (60.4-63.7) | 60.8 (59.1-62.5) | 61.1 (59.4-62.8) | +0.3 (-0.0, +0.7) | -0.9 (-2.3, +0.4) | -1.3 (-2.6, +0.1) | SAME |
| musique | zrckpn | 60.0 (58.4-61.7) | 61.8 (60.1-63.6) | 62.2 (60.5-64.0) | +0.4 (+0.0, +0.7) | +2.2 (+0.9, +3.4) | +1.8 (+0.5, +3.1) | GREATER |

## The setting's call (in-domain, both models)

| setting | call |
|---|---|
| 2wiki | SAME |
| hotpotqa | SAME |
| musique | SAME |

## zg6r - zrckpn at b1n, paired by question

| setting | read | zg6r - zrckpn (CI) | label |
|---|---|---|---|
| 2wiki | in-domain | +0.7 (+0.3, +1.0) | GNN_ABOVE |
| hotpotqa | in-domain | +0.9 (+0.4, +1.4) | GNN_ABOVE |
| hotpotqa | zero-shot | +0.8 (+0.2, +1.4) | GNN_ABOVE |
| musique | in-domain | +1.5 (+0.7, +2.3) | GNN_ABOVE |
| musique | zero-shot | -1.1 (-2.0, -0.1) | MLP_ABOVE |

## MuSiQue by hop depth (in-domain, paired by question, descriptive)

| hops | questions | model | b1n - b1w |
|---|---:|---|---|
| 2hop | 518 | zg6r | -0.1 (-0.9, +0.7) |
| 2hop | 518 | zrckpn | -0.1 (-0.8, +0.5) |
| 3hop | 316 | zg6r | -0.5 (-1.2, +0.3) |
| 3hop | 316 | zrckpn | -0.8 (-1.5, -0.1) |
| 4hop | 166 | zg6r | -0.4 (-1.2, +0.4) |
| 4hop | 166 | zrckpn | +0.5 (-0.3, +1.4) |

## The NER files

| setting | form | pairs | spaCy pairs in b1w | weight max |
|---|---|---:|---:|---:|
| 2wiki | F0T | 42247 | 36147 | 8.2857 |
| hotpotqa | F0T | 86518 | 68698 | 7.6775 |
| musique | F0T | 98845 | 75002 | 12.1533 |

## Part B: b1x (R + F3T + K), MuSiQue, descriptive, no verdict

| model | b1w | b1n | b1x | b1x - b1w | b1x - b1n | word (b1x - b1w) | word (b1x - b1n) |
|---|---:|---:|---:|---|---|---|---|
| zg6r | 69.0 (67.2-70.7) | 68.7 (67.0-70.4) | 67.2 (65.5-69.0) | -1.7 (-2.3, -1.1) | -1.5 (-2.0, -1.0) | LESS | LESS |
| zrckpn | 67.4 (65.7-69.2) | 67.2 (65.5-68.9) | 66.8 (65.1-68.5) | -0.7 (-1.2, -0.2) | -0.4 (-0.9, +0.0) | LESS | SAME |

Missing reads: 0.
