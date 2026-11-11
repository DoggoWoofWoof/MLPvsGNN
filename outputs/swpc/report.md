# SWP part C: the pair trained with S in the slot, read with S, R and nothing (N and K in every cell)

Declared in docs/SWP_S_TO_R_SWAP.md. R@5 in points, seed 0 only (GNX and zrckpn, the WQ-musique and WQ-hotpotqa fits; the in-domain regime uses the fits whose split holds the dataset, the held-out regime the others), on each dataset's 1,000-question carve. Differences are b minus a with the 95% interval of a paired bootstrap over the questions (2000 resamples); the direction is named in every call. S-trained: S in the slot while the fits train. R-trained: the same fits trained with the rule's links in the slot (Part A for MuSiQue, 2Wiki, HotpotQA; Part B for the rest; SQuAD's record slot is R already), seed 0, so the two worlds compare the same seeds. MuSiQue and SQuAD: S is itself a rule (title mentions), so there the swap is rule against rule, not rule against structure. No threshold, no adoption call.

## Reads in each slot

| dataset | regime | model | trained with | S | R | - | R vs S (R - S) | S vs none (S - none) | R vs none (R - none) |
|---|---|---|---|---:|---:|---:|---|---|---|
| musique | in-domain | gnx | S-trained | 69.3 | 69.0 | 66.3 | -0.3 (-1.1, +0.4) level | +3.1 (+1.9, +4.3) S above none | +2.8 (+1.6, +4.0) R above none |
| musique | in-domain | gnx | R-trained | 69.3 | 69.5 | 66.8 | +0.2 (-0.5, +0.8) level | +2.5 (+1.3, +3.7) S above none | +2.6 (+1.5, +3.9) R above none |
| musique | in-domain | zrckpn | S-trained | 67.3 | 67.6 | 63.9 | +0.4 (-0.3, +1.1) level | +3.3 (+2.2, +4.5) S above none | +3.7 (+2.5, +4.9) R above none |
| musique | in-domain | zrckpn | R-trained | 67.4 | 67.8 | 64.6 | +0.4 (-0.4, +1.1) level | +2.8 (+1.6, +4.1) S above none | +3.2 (+1.9, +4.5) R above none |
| musique | held-out | gnx | S-trained | 60.9 | 61.6 | 61.5 | +0.7 (-0.2, +1.7) level | -0.6 (-2.2, +1.0) level | +0.1 (-1.5, +1.8) level |
| musique | held-out | gnx | R-trained | 61.0 | 62.9 | 62.0 | +1.9 (+1.0, +2.8) R above S | -1.0 (-2.5, +0.5) level | +0.9 (-0.6, +2.3) level |
| musique | held-out | zrckpn | S-trained | 60.6 | 62.4 | 61.2 | +1.9 (+1.0, +2.7) R above S | -0.7 (-2.2, +0.8) level | +1.2 (-0.4, +2.7) level |
| musique | held-out | zrckpn | R-trained | 60.1 | 60.4 | 59.0 | +0.3 (-0.5, +1.1) level | +1.1 (-0.3, +2.5) level | +1.4 (+0.0, +2.9) R above none |
| 2wiki | in-domain | gnx | S-trained | 95.7 | 95.2 | 77.9 | -0.4 (-1.1, +0.3) level | +17.8 (+16.3, +19.1) S above none | +17.3 (+16.0, +18.8) R above none |
| 2wiki | in-domain | gnx | R-trained | 95.3 | 95.0 | 77.5 | -0.3 (-1.0, +0.4) level | +17.8 (+16.4, +19.1) S above none | +17.5 (+16.2, +18.8) R above none |
| 2wiki | in-domain | zrckpn | S-trained | 94.7 | 94.3 | 77.1 | -0.4 (-1.1, +0.3) level | +17.6 (+16.2, +19.0) S above none | +17.2 (+15.8, +18.6) R above none |
| 2wiki | in-domain | zrckpn | R-trained | 94.4 | 94.2 | 77.4 | -0.2 (-0.9, +0.5) level | +17.1 (+15.7, +18.4) S above none | +16.9 (+15.6, +18.2) R above none |
| hotpotqa | in-domain | gnx | S-trained | 96.9 | 94.0 | 89.8 | -2.9 (-3.7, -2.1) R below S | +7.0 (+5.9, +8.2) S above none | +4.2 (+3.2, +5.2) R above none |
| hotpotqa | in-domain | gnx | R-trained | 96.9 | 94.2 | 91.1 | -2.6 (-3.5, -1.9) R below S | +5.8 (+4.8, +6.9) S above none | +3.1 (+2.2, +4.1) R above none |
| hotpotqa | in-domain | zrckpn | S-trained | 96.3 | 93.5 | 90.0 | -2.9 (-3.7, -2.1) R below S | +6.3 (+5.2, +7.5) S above none | +3.5 (+2.5, +4.5) R above none |
| hotpotqa | in-domain | zrckpn | R-trained | 95.7 | 93.5 | 90.3 | -2.2 (-2.9, -1.5) R below S | +5.3 (+4.3, +6.3) S above none | +3.1 (+2.1, +4.0) R above none |
| hotpotqa | held-out | gnx | S-trained | 95.2 | 93.0 | 88.9 | -2.1 (-2.9, -1.5) R below S | +6.2 (+5.2, +7.3) S above none | +4.1 (+3.1, +5.1) R above none |
| hotpotqa | held-out | gnx | R-trained | 95.2 | 93.3 | 88.8 | -1.8 (-2.6, -1.1) R below S | +6.3 (+5.2, +7.5) S above none | +4.5 (+3.4, +5.6) R above none |
| hotpotqa | held-out | zrckpn | S-trained | 94.8 | 92.7 | 88.9 | -2.1 (-2.9, -1.4) R below S | +5.9 (+4.9, +7.0) S above none | +3.8 (+2.9, +4.8) R above none |
| hotpotqa | held-out | zrckpn | R-trained | 94.0 | 92.8 | 89.2 | -1.2 (-2.1, -0.5) R below S | +4.8 (+3.6, +5.9) S above none | +3.5 (+2.5, +4.5) R above none |
| squad | in-domain | gnx | S-trained | 91.1 | 91.1 | 91.2 | -0.0 (-0.2, +0.1) level | -0.1 (-0.3, +0.0) level | -0.2 (-0.3, -0.0) R below none |
| squad | in-domain | gnx | R-trained | 91.0 | 91.5 | 91.0 | +0.5 (+0.3, +0.7) R above S | -0.1 (-0.2, +0.1) level | +0.4 (+0.3, +0.6) R above none |
| squad | in-domain | zrckpn | S-trained | 91.2 | 91.1 | 91.3 | -0.1 (-0.2, +0.1) level | -0.1 (-0.2, +0.0) level | -0.2 (-0.3, -0.0) R below none |
| squad | in-domain | zrckpn | R-trained | 91.1 | 91.3 | 91.3 | +0.3 (+0.1, +0.4) R above S | -0.2 (-0.4, -0.1) S below none | +0.1 (-0.1, +0.2) level |
| metaqa | held-out | gnx | S-trained | 35.6 | 2.3 | 0.5 | -33.3 (-34.1, -32.5) R below S | +35.2 (+34.3, +36.0) S above none | +1.8 (+1.6, +2.0) R above none |
| metaqa | held-out | gnx | R-trained | 37.5 | 9.7 | 0.6 | -27.8 (-28.5, -27.1) R below S | +36.9 (+36.1, +37.7) S above none | +9.1 (+8.7, +9.5) R above none |
| metaqa | held-out | zrckpn | S-trained | 38.0 | 1.1 | 0.7 | -36.8 (-37.7, -36.1) R below S | +37.3 (+36.5, +38.1) S above none | +0.5 (+0.4, +0.6) R above none |
| metaqa | held-out | zrckpn | R-trained | 39.0 | 1.3 | 0.7 | -37.7 (-38.5, -36.9) R below S | +38.3 (+37.5, +39.1) S above none | +0.6 (+0.5, +0.8) R above none |
| webqsp | in-domain | gnx | S-trained | 66.6 | 11.4 | 8.3 | -55.2 (-57.3, -53.0) R below S | +58.4 (+56.1, +60.4) S above none | +3.2 (+2.3, +4.0) R above none |
| webqsp | in-domain | gnx | R-trained | 66.4 | 10.6 | 7.8 | -55.7 (-57.8, -53.5) R below S | +58.5 (+56.2, +60.6) S above none | +2.8 (+1.9, +3.7) R above none |
| webqsp | in-domain | zrckpn | S-trained | 64.9 | 10.4 | 7.8 | -54.4 (-56.6, -52.1) R below S | +57.1 (+54.9, +59.4) S above none | +2.7 (+1.9, +3.5) R above none |
| webqsp | in-domain | zrckpn | R-trained | 64.6 | 10.6 | 8.2 | -54.0 (-56.2, -51.7) R below S | +56.4 (+54.2, +58.6) S above none | +2.4 (+1.6, +3.2) R above none |

## Does the S slot in training change what a read gains from S? Same slot, S-trained minus R-trained

| dataset | regime | slot read | model | R-trained | S-trained | S-trained - R-trained |
|---|---|---|---|---:|---:|---|
| musique | in-domain | S | gnx | 69.3 | 69.3 | +0.1 (-0.8, +1.0) level |
| musique | in-domain | S | zrckpn | 67.4 | 67.3 | -0.1 (-1.1, +0.7) level |
| musique | in-domain | R | gnx | 69.5 | 69.0 | -0.4 (-1.4, +0.4) level |
| musique | in-domain | R | zrckpn | 67.8 | 67.6 | -0.2 (-1.0, +0.7) level |
| musique | in-domain | - | gnx | 66.8 | 66.3 | -0.5 (-1.5, +0.4) level |
| musique | in-domain | - | zrckpn | 64.6 | 63.9 | -0.7 (-1.5, +0.3) level |
| musique | held-out | S | gnx | 61.0 | 60.9 | -0.1 (-1.2, +1.0) level |
| musique | held-out | S | zrckpn | 60.1 | 60.6 | +0.5 (-0.5, +1.5) level |
| musique | held-out | R | gnx | 62.9 | 61.6 | -1.3 (-2.4, -0.2) S-trained below R-trained |
| musique | held-out | R | zrckpn | 60.4 | 62.4 | +2.0 (+1.1, +2.9) S-trained above R-trained |
| musique | held-out | - | gnx | 62.0 | 61.5 | -0.5 (-1.6, +0.6) level |
| musique | held-out | - | zrckpn | 59.0 | 61.2 | +2.2 (+1.2, +3.2) S-trained above R-trained |
| 2wiki | in-domain | S | gnx | 95.3 | 95.7 | +0.3 (+0.0, +0.6) S-trained above R-trained |
| 2wiki | in-domain | S | zrckpn | 94.4 | 94.7 | +0.3 (-0.1, +0.6) level |
| 2wiki | in-domain | R | gnx | 95.0 | 95.2 | +0.2 (-0.1, +0.5) level |
| 2wiki | in-domain | R | zrckpn | 94.2 | 94.3 | +0.0 (-0.4, +0.4) level |
| 2wiki | in-domain | - | gnx | 77.5 | 77.9 | +0.3 (-0.2, +0.9) level |
| 2wiki | in-domain | - | zrckpn | 77.4 | 77.1 | -0.3 (-0.7, +0.2) level |
| hotpotqa | in-domain | S | gnx | 96.9 | 96.9 | -0.1 (-0.5, +0.4) level |
| hotpotqa | in-domain | S | zrckpn | 95.7 | 96.3 | +0.7 (+0.1, +1.2) S-trained above R-trained |
| hotpotqa | in-domain | R | gnx | 94.2 | 94.0 | -0.2 (-0.8, +0.3) level |
| hotpotqa | in-domain | R | zrckpn | 93.5 | 93.5 | +0.0 (-0.6, +0.5) level |
| hotpotqa | in-domain | - | gnx | 91.1 | 89.8 | -1.3 (-2.1, -0.4) S-trained below R-trained |
| hotpotqa | in-domain | - | zrckpn | 90.3 | 90.0 | -0.4 (-1.1, +0.3) level |
| hotpotqa | held-out | S | gnx | 95.2 | 95.2 | +0.0 (-0.7, +0.6) level |
| hotpotqa | held-out | S | zrckpn | 94.0 | 94.8 | +0.9 (+0.2, +1.5) S-trained above R-trained |
| hotpotqa | held-out | R | gnx | 93.3 | 93.0 | -0.3 (-1.0, +0.4) level |
| hotpotqa | held-out | R | zrckpn | 92.8 | 92.7 | -0.1 (-0.8, +0.7) level |
| hotpotqa | held-out | - | gnx | 88.8 | 88.9 | +0.1 (-0.9, +1.1) level |
| hotpotqa | held-out | - | zrckpn | 89.2 | 88.9 | -0.4 (-1.2, +0.5) level |
| squad | in-domain | S | gnx | 91.0 | 91.1 | +0.1 (-0.1, +0.3) level |
| squad | in-domain | S | zrckpn | 91.1 | 91.2 | +0.1 (-0.1, +0.2) level |
| squad | in-domain | R | gnx | 91.5 | 91.1 | -0.4 (-0.6, -0.2) S-trained below R-trained |
| squad | in-domain | R | zrckpn | 91.4 | 91.1 | -0.2 (-0.4, -0.1) S-trained below R-trained |
| squad | in-domain | - | gnx | 91.0 | 91.2 | +0.2 (+0.0, +0.4) S-trained above R-trained |
| squad | in-domain | - | zrckpn | 91.3 | 91.3 | -0.0 (-0.2, +0.1) level |
| metaqa | held-out | S | gnx | 37.4 | 35.6 | -1.8 (-2.2, -1.4) S-trained below R-trained |
| metaqa | held-out | S | zrckpn | 39.0 | 37.9 | -1.0 (-1.5, -0.6) S-trained below R-trained |
| metaqa | held-out | R | gnx | 9.7 | 2.3 | -7.4 (-7.7, -7.0) S-trained below R-trained |
| metaqa | held-out | R | zrckpn | 1.3 | 1.1 | -0.2 (-0.3, -0.0) S-trained below R-trained |
| metaqa | held-out | - | gnx | 0.6 | 0.5 | -0.1 (-0.2, -0.0) S-trained below R-trained |
| metaqa | held-out | - | zrckpn | 0.7 | 0.6 | -0.0 (-0.1, +0.0) level |
| webqsp | in-domain | S | gnx | 66.4 | 66.6 | +0.3 (-0.4, +1.0) level |
| webqsp | in-domain | S | zrckpn | 64.6 | 64.9 | +0.3 (-0.4, +0.9) level |
| webqsp | in-domain | R | gnx | 10.6 | 11.4 | +0.8 (+0.1, +1.5) S-trained above R-trained |
| webqsp | in-domain | R | zrckpn | 10.6 | 10.4 | -0.2 (-0.9, +0.5) level |
| webqsp | in-domain | - | gnx | 7.8 | 8.3 | +0.4 (-0.1, +0.9) level |
| webqsp | in-domain | - | zrckpn | 8.2 | 7.8 | -0.5 (-1.1, +0.2) level |

## Do we need message passing? GNX minus zrckpn on the same inputs, in each slot

| dataset | regime | trained with | S | R | - | gap change R vs S | gap change S vs none |
|---|---|---|---|---|---|---|---|
| musique | in-domain | S-trained | +2.1 (+1.1, +3.0) GNX above zrckpn | +1.4 (+0.4, +2.4) GNX above zrckpn | +2.4 (+1.3, +3.3) GNX above zrckpn | -0.7 (-1.6, +0.2) level | -0.3 (-1.5, +1.0) level |
| musique | in-domain | R-trained | +1.8 (+0.8, +2.9) GNX above zrckpn | +1.7 (+0.7, +2.7) GNX above zrckpn | +2.2 (+1.3, +3.3) GNX above zrckpn | -0.2 (-1.1, +0.8) level | -0.4 (-1.8, +1.0) level |
| musique | held-out | S-trained | +0.3 (-0.8, +1.4) level | -0.8 (-2.1, +0.4) level | +0.3 (-0.7, +1.2) level | -1.1 (-2.3, -0.0) R slot below S slot | +0.1 (-1.4, +1.5) level |
| musique | held-out | R-trained | +0.9 (-0.2, +2.0) level | +2.5 (+1.4, +3.6) GNX above zrckpn | +3.0 (+1.8, +4.1) GNX above zrckpn | +1.6 (+0.6, +2.5) R slot above S slot | -2.1 (-3.4, -0.7) S slot below none |
| 2wiki | in-domain | S-trained | +0.9 (+0.6, +1.3) GNX above zrckpn | +1.0 (+0.6, +1.4) GNX above zrckpn | +0.8 (+0.3, +1.3) GNX above zrckpn | +0.0 (-0.4, +0.4) level | +0.2 (-0.4, +0.7) level |
| 2wiki | in-domain | R-trained | +0.9 (+0.5, +1.3) GNX above zrckpn | +0.8 (+0.4, +1.1) GNX above zrckpn | +0.2 (-0.3, +0.7) level | -0.1 (-0.5, +0.3) level | +0.7 (+0.1, +1.3) S slot above none |
| hotpotqa | in-domain | S-trained | +0.5 (+0.1, +1.1) GNX above zrckpn | +0.5 (-0.1, +1.2) level | -0.1 (-0.9, +0.5) level | +0.0 (-0.7, +0.7) level | +0.7 (-0.1, +1.6) level |
| hotpotqa | in-domain | R-trained | +1.2 (+0.7, +1.8) GNX above zrckpn | +0.8 (+0.2, +1.4) GNX above zrckpn | +0.8 (+0.0, +1.5) level | -0.4 (-1.1, +0.2) level | +0.5 (-0.4, +1.4) level |
| hotpotqa | held-out | S-trained | +0.4 (-0.1, +0.9) level | +0.4 (-0.3, +1.0) level | +0.1 (-0.7, +0.8) level | +0.0 (-0.7, +0.7) level | +0.3 (-0.5, +1.1) level |
| hotpotqa | held-out | R-trained | +1.2 (+0.5, +1.9) GNX above zrckpn | +0.6 (-0.1, +1.3) level | -0.4 (-1.4, +0.5) level | -0.6 (-1.4, +0.1) level | +1.6 (+0.5, +2.7) S slot above none |
| squad | in-domain | S-trained | -0.1 (-0.2, +0.1) level | -0.1 (-0.2, +0.1) level | -0.1 (-0.2, +0.1) level | +0.0 (-0.1, +0.2) level | -0.0 (-0.2, +0.1) level |
| squad | in-domain | R-trained | -0.1 (-0.3, +0.0) level | +0.1 (-0.0, +0.3) level | -0.3 (-0.4, -0.1) GNX below zrckpn | +0.2 (+0.1, +0.4) R slot above S slot | +0.1 (-0.0, +0.3) level |
| metaqa | held-out | S-trained | -2.3 (-2.9, -1.8) GNX below zrckpn | +1.2 (+1.0, +1.3) GNX above zrckpn | -0.2 (-0.3, -0.1) GNX below zrckpn | +3.5 (+2.9, +4.1) R slot above S slot | -2.1 (-2.7, -1.6) S slot below none |
| metaqa | held-out | R-trained | -1.5 (-2.1, -0.9) GNX below zrckpn | +8.4 (+8.0, +8.8) GNX above zrckpn | -0.1 (-0.2, -0.0) GNX below zrckpn | +9.9 (+9.2, +10.7) R slot above S slot | -1.4 (-2.0, -0.8) S slot below none |
| webqsp | in-domain | S-trained | +1.7 (+0.5, +2.9) GNX above zrckpn | +1.0 (+0.3, +1.6) GNX above zrckpn | +0.5 (-0.1, +1.1) level | -0.8 (-2.1, +0.6) level | +1.2 (-0.1, +2.6) level |
| webqsp | in-domain | R-trained | +1.7 (+0.3, +3.0) GNX above zrckpn | -0.0 (-0.7, +0.8) level | -0.4 (-1.0, +0.3) level | -1.7 (-3.2, -0.2) R slot below S slot | +2.1 (+0.5, +3.5) S slot above none |

## Reference only: B1L's LLM-built P edges (carve b1o) beside the slots, zrckpn trained with R in the slot, seed 0

B1L (568bf08) read the models of record on these edges; GNX was not read on them, so there is no S-trained or GNX cell. The b1o entries are NULL-heavy (most have no W2_S type) while the fits were trained on W2_S types, so b1o is a lower bound for an LLM-built graph. This block says nothing about which graph is better.

| dataset | regime | b1t | b1w (R) | b1 (S) | b1o (LLM edges) | b1o vs none (b1o - b1t) | b1o vs R (b1o - b1w) | b1o vs S (b1o - b1) |
|---|---|---:|---:|---:|---:|---|---|---|
| musique | in-domain | 64.6 | 67.8 | 67.4 | 68.1 | +3.5 (+2.1, +4.9) b1o above b1t | +0.3 (-0.9, +1.5) level | +0.7 (-0.6, +1.8) level |
| musique | held-out | 59.0 | 60.4 | 60.1 | 62.6 | +3.5 (+2.1, +5.2) b1o above b1t | +2.1 (+0.6, +3.7) b1o above b1w | +2.5 (+0.9, +4.1) b1o above b1 |
| 2wiki | in-domain | 77.4 | 94.2 | 94.4 | 91.0 | +13.6 (+12.3, +15.0) b1o above b1t | -3.3 (-4.3, -2.3) b1o below b1w | -3.5 (-4.5, -2.5) b1o below b1 |
