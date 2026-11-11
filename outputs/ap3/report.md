# AP3 report, Part B (declared in docs/AP3_STRUCTURAL_SLOT_TITLE_FORMS.md)

R@5 x100 on HippoRAG 2's settings (1,000 questions each), no new fit. The models of record (zg6r, zrckpn) are read on b1t (N + K), b1w (R + spaCy's N + K), b1n (R + F0T + K, AP2) and **b1y (R + the carried rung's base-title pairs in the structural slot, F0T, K)**: the other files are b1n's. A read-time shift: the models were fitted with R in the structural slot. In-domain = the mean over the fits that trained on the setting's partner split. 95% bootstrap intervals over the questions (2,000 resamples, seed 0); words by the interval, no threshold (GREATER / SAME / LESS for b1y - b1n and b1y - b1w; UP / DOWN / FLAT against b1t).

**Verdict (Part B): HOLD** (carried rungs: 2wiki None, hotpotqa 3, musique 25).

## In-domain

| setting | model | b1t | b1w | b1n | b1y | b1y - b1n | b1y - b1w | b1y - b1t | word (b1y - b1n) |
|---|---|---:|---:|---:|---:|---|---|---|---|
| hotpotqa | zg6r | 91.1 (90.0-92.3) | 94.3 (93.3-95.3) | -- | 95.1 (94.1-96.0) | +0.8 (+0.4, +1.2) | +0.8 (+0.3, +1.3) | +4.0 (+3.1, +4.8) | GREATER |
| hotpotqa | zrckpn | 90.4 (89.2-91.6) | 93.4 (92.4-94.5) | -- | 94.2 (93.2-95.2) | +0.8 (+0.4, +1.1) | +0.8 (+0.4, +1.2) | +3.9 (+3.0, +4.8) | GREATER |
| musique | zg6r | 66.0 (64.3-67.7) | 69.0 (67.2-70.7) | -- | 68.6 (66.9-70.3) | -0.1 (-0.5, +0.3) | -0.4 (-0.9, +0.2) | +2.6 (+1.5, +3.8) | SAME |
| musique | zrckpn | 64.5 (62.8-66.1) | 67.4 (65.7-69.2) | -- | 66.7 (65.0-68.4) | -0.5 (-1.0, -0.1) | -0.7 (-1.3, -0.2) | +2.2 (+1.0, +3.3) | LESS |

## Held-out

| setting | model | b1t | b1w | b1n | b1y | b1y - b1n | b1y - b1w | b1y - b1t | word (b1y - b1n) |
|---|---|---:|---:|---:|---:|---|---|---|---|
| hotpotqa | zg6r | 89.4 (88.2-90.6) | 92.9 (91.9-94.0) | -- | 93.5 (92.5-94.5) | +0.6 (+0.3, +1.0) | +0.6 (+0.1, +1.0) | +4.1 (+3.2, +5.0) | GREATER |
| hotpotqa | zrckpn | 88.6 (87.4-89.9) | 92.2 (91.2-93.4) | -- | 92.7 (91.6-93.8) | +0.7 (+0.3, +1.0) | +0.5 (+0.1, +0.9) | +4.1 (+3.2, +5.0) | GREATER |
| musique | zg6r | 62.1 (60.4-63.7) | 60.8 (59.1-62.5) | -- | 61.2 (59.5-62.9) | +0.1 (-0.4, +0.7) | +0.4 (-0.2, +1.1) | -0.8 (-2.2, +0.5) | SAME |
| musique | zrckpn | 60.0 (58.4-61.7) | 61.8 (60.1-63.6) | -- | 62.0 (60.3-63.7) | -0.1 (-0.7, +0.4) | +0.2 (-0.4, +0.8) | +2.0 (+0.7, +3.3) | SAME |

## The setting's call (in-domain, both models)

| setting | call |
|---|---|
| hotpotqa | GREATER |
| musique | MIXED |

## zg6r - zrckpn at b1y, paired by question

| setting | read | zg6r - zrckpn (CI) | label |
|---|---|---|---|
| hotpotqa | in-domain | +0.9 (+0.4, +1.3) | GNN_ABOVE |
| hotpotqa | zero-shot | +0.7 (+0.2, +1.3) | GNN_ABOVE |
| musique | in-domain | +1.9 (+1.1, +2.8) | GNN_ABOVE |
| musique | zero-shot | -0.8 (-1.7, +0.1) | LEVEL |

## MuSiQue by hop depth (in-domain, paired by question, descriptive)

| hops | questions | model | b1y - b1n |
|---|---:|---|---|
| 2hop | 518 | zg6r | +0.1 (-0.6, +0.8) |
| 2hop | 518 | zrckpn | -0.4 (-1.1, +0.3) |
| 3hop | 316 | zg6r | -0.6 (-1.2, +0.1) |
| 3hop | 316 | zrckpn | -1.0 (-1.9, -0.2) |
| 4hop | 166 | zg6r | +0.3 (-0.4, +1.1) |
| 4hop | 166 | zrckpn | +0.1 (-0.5, +0.9) |

## The structural files

| setting | df cap | R pairs | pairs added | pairs in the carve |
|---|---|---:|---:|---:|
| hotpotqa | 3 | 4315 | 357 | 4672 |
| musique | 25 | 41429 | 1864 | 43293 |

Missing reads: 0.
