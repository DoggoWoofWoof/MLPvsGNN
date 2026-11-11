# SWP part E: the pair trained with the slot EMPTY (N and K only), read with nothing, R and S; beside the R-trained and S-trained pair

Declared in docs/SWP_S_TO_R_SWAP.md (section 5f). R@5 in points, seed 0 only (GNX and zrckpn, the WQ-musique and WQ-hotpotqa fits; the in-domain regime uses the fits whose split holds the dataset, the held-out regime the others), on each dataset's carve. Differences are b minus a with the 95% interval of a paired bootstrap over the questions (2000 resamples, one mean per resample); the direction is named in every call. N-trained: the slot empty while the fits train (Part E). R-trained: the rule's links in the slot (Parts A and B; SQuAD's record slot is R already). S-trained: the dataset's own S in the slot (Part C). N (rule NER links) and K (kNN) are in every cell, so 'empty' means no structural slot, not no graph. MuSiQue and SQuAD: S is itself a rule (title mentions). No threshold, no adoption call.

## Reads in each slot

| dataset | regime | model | trained with | - | R | S |
|---|---|---|---|---:|---:|---:|
| musique | in-domain | gnx | N-trained | 66.5 | 67.0 | 66.9 |
| musique | in-domain | gnx | R-trained | 66.8 | 69.5 | 69.3 |
| musique | in-domain | gnx | S-trained | 66.3 | 69.0 | 69.3 |
| musique | in-domain | zrckpn | N-trained | 64.8 | 65.7 | 65.3 |
| musique | in-domain | zrckpn | R-trained | 64.6 | 67.8 | 67.4 |
| musique | in-domain | zrckpn | S-trained | 63.9 | 67.6 | 67.3 |
| musique | held-out | gnx | N-trained | 50.8 | 60.8 | 60.8 |
| musique | held-out | gnx | R-trained | 62.0 | 62.9 | 61.0 |
| musique | held-out | gnx | S-trained | 61.5 | 61.6 | 60.9 |
| musique | held-out | zrckpn | N-trained | 57.7 | 60.6 | 60.3 |
| musique | held-out | zrckpn | R-trained | 59.0 | 60.4 | 60.1 |
| musique | held-out | zrckpn | S-trained | 61.2 | 62.4 | 60.6 |
| 2wiki | in-domain | gnx | N-trained | 78.1 | 90.4 | 90.7 |
| 2wiki | in-domain | gnx | R-trained | 77.5 | 95.0 | 95.3 |
| 2wiki | in-domain | gnx | S-trained | 77.9 | 95.2 | 95.7 |
| 2wiki | in-domain | zrckpn | N-trained | 77.3 | 89.2 | 89.5 |
| 2wiki | in-domain | zrckpn | R-trained | 77.4 | 94.2 | 94.4 |
| 2wiki | in-domain | zrckpn | S-trained | 77.1 | 94.3 | 94.7 |
| hotpotqa | in-domain | gnx | N-trained | 90.3 | 91.8 | 93.2 |
| hotpotqa | in-domain | gnx | R-trained | 91.1 | 94.2 | 96.9 |
| hotpotqa | in-domain | gnx | S-trained | 89.8 | 94.0 | 96.9 |
| hotpotqa | in-domain | zrckpn | N-trained | 90.1 | 92.0 | 93.9 |
| hotpotqa | in-domain | zrckpn | R-trained | 90.3 | 93.5 | 95.7 |
| hotpotqa | in-domain | zrckpn | S-trained | 90.0 | 93.5 | 96.3 |
| hotpotqa | held-out | gnx | N-trained | 89.3 | 92.5 | 93.8 |
| hotpotqa | held-out | gnx | R-trained | 88.8 | 93.3 | 95.2 |
| hotpotqa | held-out | gnx | S-trained | 88.9 | 93.0 | 95.2 |
| hotpotqa | held-out | zrckpn | N-trained | 89.5 | 91.8 | 93.2 |
| hotpotqa | held-out | zrckpn | R-trained | 89.2 | 92.8 | 94.0 |
| hotpotqa | held-out | zrckpn | S-trained | 88.9 | 92.7 | 94.8 |
| squad | in-domain | gnx | N-trained | 91.3 | 91.2 | 91.2 |
| squad | in-domain | gnx | R-trained | 91.0 | 91.5 | 91.0 |
| squad | in-domain | gnx | S-trained | 91.2 | 91.1 | 91.1 |
| squad | in-domain | zrckpn | N-trained | 91.3 | 91.2 | 91.2 |
| squad | in-domain | zrckpn | R-trained | 91.3 | 91.3 | 91.1 |
| squad | in-domain | zrckpn | S-trained | 91.3 | 91.1 | 91.2 |
| metaqa | held-out | gnx | N-trained | 0.9 | 1.9 | 5.1 |
| metaqa | held-out | gnx | R-trained | 0.6 | 9.7 | 37.5 |
| metaqa | held-out | gnx | S-trained | 0.5 | 2.3 | 35.6 |
| metaqa | held-out | zrckpn | N-trained | 1.0 | 1.1 | 3.8 |
| metaqa | held-out | zrckpn | R-trained | 0.7 | 1.3 | 39.0 |
| metaqa | held-out | zrckpn | S-trained | 0.7 | 1.1 | 38.0 |
| webqsp | in-domain | gnx | N-trained | 27.4 | 31.5 | 32.5 |
| webqsp | in-domain | gnx | R-trained | 7.8 | 10.6 | 66.4 |
| webqsp | in-domain | gnx | S-trained | 8.3 | 11.4 | 66.6 |
| webqsp | in-domain | zrckpn | N-trained | 26.8 | 28.8 | 28.1 |
| webqsp | in-domain | zrckpn | R-trained | 8.2 | 10.6 | 64.6 |
| webqsp | in-domain | zrckpn | S-trained | 7.8 | 10.4 | 64.9 |

## What the slot adds when the model trains with it (the diagonal), and the read-time removal beside it

| dataset | regime | model | N-trained, empty | R-trained, R | S-trained, S | R-trained R - N-trained empty | S-trained S - N-trained empty | N-trained empty - R-trained empty |
|---|---|---|---:|---:|---:|---|---|---|
| musique | in-domain | gnx | 66.5 | 69.5 | 69.3 | +2.9 (+1.6, +4.2) R-trained R above N-trained empty | +2.8 (+1.5, +4.1) S-trained S above N-trained empty | -0.3 (-1.3, +0.8) level |
| musique | in-domain | zrckpn | 64.8 | 67.8 | 67.3 | +3.0 (+1.8, +4.2) R-trained R above N-trained empty | +2.5 (+1.3, +3.7) S-trained S above N-trained empty | +0.2 (-0.9, +1.4) level |
| musique | held-out | gnx | 50.8 | 62.9 | 60.9 | +12.1 (+10.3, +14.0) R-trained R above N-trained empty | +10.1 (+8.3, +12.1) S-trained S above N-trained empty | -11.2 (-13.0, -9.5) N-trained empty below R-trained empty |
| musique | held-out | zrckpn | 57.7 | 60.4 | 60.6 | +2.7 (+0.9, +4.4) R-trained R above N-trained empty | +2.9 (+1.1, +4.6) S-trained S above N-trained empty | -1.3 (-3.0, +0.3) level |
| 2wiki | in-domain | gnx | 78.1 | 95.0 | 95.7 | +16.9 (+15.5, +18.2) R-trained R above N-trained empty | +17.5 (+16.1, +18.8) S-trained S above N-trained empty | +0.6 (+0.0, +1.2) N-trained empty above R-trained empty |
| 2wiki | in-domain | zrckpn | 77.3 | 94.2 | 94.7 | +16.9 (+15.6, +18.3) R-trained R above N-trained empty | +17.4 (+15.9, +18.7) S-trained S above N-trained empty | -0.1 (-0.7, +0.6) level |
| hotpotqa | in-domain | gnx | 90.3 | 94.2 | 96.9 | +4.0 (+2.9, +5.0) R-trained R above N-trained empty | +6.6 (+5.4, +7.8) S-trained S above N-trained empty | -0.8 (-1.7, +0.1) level |
| hotpotqa | in-domain | zrckpn | 90.1 | 93.5 | 96.3 | +3.4 (+2.4, +4.4) R-trained R above N-trained empty | +6.2 (+5.1, +7.2) S-trained S above N-trained empty | -0.2 (-1.0, +0.5) level |
| hotpotqa | held-out | gnx | 89.3 | 93.3 | 95.2 | +4.0 (+2.9, +5.1) R-trained R above N-trained empty | +5.9 (+4.8, +7.0) S-trained S above N-trained empty | +0.5 (-0.5, +1.5) level |
| hotpotqa | held-out | zrckpn | 89.5 | 92.8 | 94.8 | +3.2 (+2.1, +4.2) R-trained R above N-trained empty | +5.3 (+4.2, +6.4) S-trained S above N-trained empty | +0.3 (-0.6, +1.2) level |
| squad | in-domain | gnx | 91.3 | 91.5 | 91.1 | +0.2 (+0.0, +0.4) R-trained R above N-trained empty | -0.2 (-0.4, -0.0) S-trained S below N-trained empty | +0.2 (+0.1, +0.4) N-trained empty above R-trained empty |
| squad | in-domain | zrckpn | 91.3 | 91.3 | 91.2 | +0.0 (-0.1, +0.2) level | -0.2 (-0.3, +0.0) level | +0.0 (-0.1, +0.2) level |
| metaqa | held-out | gnx | 0.9 | 9.7 | 35.6 | +8.7 (+8.3, +9.1) R-trained R above N-trained empty | +34.7 (+33.9, +35.5) S-trained S above N-trained empty | +0.4 (+0.3, +0.5) N-trained empty above R-trained empty |
| metaqa | held-out | zrckpn | 1.0 | 1.3 | 38.0 | +0.2 (+0.1, +0.4) R-trained R above N-trained empty | +36.9 (+36.1, +37.7) S-trained S above N-trained empty | +0.4 (+0.3, +0.5) N-trained empty above R-trained empty |
| webqsp | in-domain | gnx | 27.4 | 10.6 | 66.6 | -16.7 (-18.6, -14.8) R-trained R below N-trained empty | +39.3 (+37.0, +41.5) S-trained S above N-trained empty | +19.5 (+17.8, +21.2) N-trained empty above R-trained empty |
| webqsp | in-domain | zrckpn | 26.8 | 10.6 | 64.9 | -16.1 (-17.9, -14.3) R-trained R below N-trained empty | +38.1 (+35.6, +40.5) S-trained S above N-trained empty | +18.5 (+16.8, +20.2) N-trained empty above R-trained empty |

## Do we need message passing? GNX minus zrckpn on the same inputs

| dataset | regime | N-trained, empty | N-trained, R | N-trained, S | R-trained, R | S-trained, S | gap change R-trained R vs N-trained empty | gap change S-trained S vs N-trained empty |
|---|---|---|---|---|---|---|---|---|
| musique | in-domain | +1.8 (+0.7, +2.7) GNX above zrckpn | +1.3 (+0.3, +2.3) GNX above zrckpn | +1.6 (+0.6, +2.5) GNX above zrckpn | +1.7 (+0.7, +2.7) GNX above zrckpn | +2.1 (+1.1, +3.0) GNX above zrckpn | -0.1 (-1.4, +1.4) level | +0.3 (-1.0, +1.7) level |
| musique | held-out | -7.0 (-8.4, -5.5) GNX below zrckpn | +0.2 (-1.0, +1.4) level | +0.5 (-0.8, +1.7) level | +2.5 (+1.4, +3.6) GNX above zrckpn | +0.3 (-0.8, +1.4) level | +9.4 (+7.5, +11.2) R-trained R above N-trained empty | +7.3 (+5.4, +9.2) S-trained S above N-trained empty |
| 2wiki | in-domain | +0.8 (+0.4, +1.3) GNX above zrckpn | +1.2 (+0.6, +1.7) GNX above zrckpn | +1.2 (+0.6, +1.8) GNX above zrckpn | +0.8 (+0.4, +1.1) GNX above zrckpn | +0.9 (+0.6, +1.3) GNX above zrckpn | -0.1 (-0.7, +0.5) level | +0.1 (-0.5, +0.7) level |
| hotpotqa | in-domain | +0.2 (-0.6, +0.9) level | -0.2 (-1.1, +0.5) level | -0.7 (-1.4, +0.1) level | +0.8 (+0.2, +1.4) GNX above zrckpn | +0.5 (+0.1, +1.1) GNX above zrckpn | +0.6 (-0.4, +1.6) level | +0.4 (-0.7, +1.3) level |
| hotpotqa | held-out | -0.2 (-1.0, +0.6) level | +0.7 (-0.1, +1.4) level | +0.5 (-0.1, +1.3) level | +0.6 (-0.1, +1.3) level | +0.4 (-0.1, +0.9) level | +0.8 (-0.2, +1.8) level | +0.5 (-0.4, +1.6) level |
| squad | in-domain | -0.0 (-0.2, +0.1) level | -0.0 (-0.2, +0.2) level | +0.0 (-0.2, +0.2) level | +0.1 (-0.0, +0.3) level | -0.1 (-0.2, +0.1) level | +0.2 (-0.0, +0.4) level | -0.0 (-0.2, +0.2) level |
| metaqa | held-out | -0.1 (-0.2, +0.0) level | +0.9 (+0.7, +1.0) GNX above zrckpn | +1.3 (+1.0, +1.5) GNX above zrckpn | +8.4 (+8.0, +8.8) GNX above zrckpn | -2.3 (-2.9, -1.8) GNX below zrckpn | +8.5 (+8.1, +8.9) R-trained R above N-trained empty | -2.2 (-2.8, -1.7) S-trained S below N-trained empty |
| webqsp | in-domain | +0.6 (-0.2, +1.5) level | +2.6 (+1.6, +3.7) GNX above zrckpn | +4.4 (+3.3, +5.5) GNX above zrckpn | -0.0 (-0.7, +0.8) level | +1.7 (+0.5, +2.9) GNX above zrckpn | -0.6 (-1.8, +0.5) level | +1.1 (-0.4, +2.7) level |

## Beside, no claim: transfer to a slot the fit never trained with

| dataset | regime | model | N-trained R - N-trained empty | N-trained S - N-trained empty | R-trained R - N-trained R | S-trained S - N-trained S |
|---|---|---|---|---|---|---|
| musique | in-domain | gnx | +0.5 (-0.6, +1.6) level | +0.4 (-0.6, +1.5) level | +2.5 (+1.3, +3.6) R-trained above N-trained | +2.5 (+1.3, +3.5) S-trained above N-trained |
| musique | in-domain | zrckpn | +0.9 (-0.2, +2.0) level | +0.6 (-0.6, +1.7) level | +2.1 (+1.0, +3.2) R-trained above N-trained | +1.9 (+0.9, +3.0) S-trained above N-trained |
| musique | held-out | gnx | +10.0 (+8.6, +11.4) R above empty | +10.0 (+8.6, +11.4) S above empty | +2.1 (+0.5, +3.8) R-trained above N-trained | +0.1 (-1.5, +1.8) level |
| musique | held-out | zrckpn | +2.9 (+1.7, +4.0) R above empty | +2.6 (+1.5, +3.8) S above empty | -0.2 (-1.7, +1.4) level | +0.3 (-1.2, +1.9) level |
| 2wiki | in-domain | gnx | +12.2 (+11.1, +13.4) R above empty | +12.6 (+11.4, +13.7) S above empty | +4.7 (+4.0, +5.3) R-trained above N-trained | +4.9 (+4.2, +5.6) S-trained above N-trained |
| 2wiki | in-domain | zrckpn | +11.9 (+10.8, +13.0) R above empty | +12.2 (+11.1, +13.3) S above empty | +5.1 (+4.3, +5.8) R-trained above N-trained | +5.2 (+4.4, +6.0) S-trained above N-trained |
| hotpotqa | in-domain | gnx | +1.5 (+0.9, +2.2) R above empty | +2.9 (+2.2, +3.7) S above empty | +2.5 (+1.7, +3.4) R-trained above N-trained | +3.6 (+2.8, +4.7) S-trained above N-trained |
| hotpotqa | in-domain | zrckpn | +1.9 (+1.1, +2.8) R above empty | +3.8 (+2.9, +4.7) S above empty | +1.4 (+0.7, +2.1) R-trained above N-trained | +2.4 (+1.7, +3.2) S-trained above N-trained |
| hotpotqa | held-out | gnx | +3.1 (+2.2, +4.0) R above empty | +4.5 (+3.5, +5.5) S above empty | +0.9 (+0.1, +1.7) R-trained above N-trained | +1.4 (+0.8, +2.1) S-trained above N-trained |
| hotpotqa | held-out | zrckpn | +2.2 (+1.3, +3.2) R above empty | +3.7 (+2.7, +4.7) S above empty | +0.9 (+0.1, +1.8) R-trained above N-trained | +1.6 (+0.8, +2.4) S-trained above N-trained |
| squad | in-domain | gnx | -0.1 (-0.2, +0.1) level | -0.1 (-0.2, +0.0) level | +0.3 (+0.1, +0.5) R-trained above N-trained | -0.1 (-0.3, +0.1) level |
| squad | in-domain | zrckpn | -0.1 (-0.2, +0.0) level | -0.1 (-0.2, -0.0) S below empty | +0.1 (-0.0, +0.3) level | -0.0 (-0.2, +0.2) level |
| metaqa | held-out | gnx | +1.0 (+0.7, +1.2) R above empty | +4.2 (+3.8, +4.5) S above empty | +7.8 (+7.3, +8.2) R-trained above N-trained | +30.5 (+29.8, +31.3) S-trained above N-trained |
| metaqa | held-out | zrckpn | +0.0 (-0.1, +0.2) level | +2.8 (+2.5, +3.1) S above empty | +0.2 (+0.0, +0.4) R-trained above N-trained | +34.1 (+33.3, +34.9) S-trained above N-trained |
| webqsp | in-domain | gnx | +4.1 (+3.0, +5.3) R above empty | +5.1 (+3.7, +6.5) S above empty | -20.8 (-22.7, -19.0) R-trained below N-trained | +34.1 (+31.9, +36.3) S-trained above N-trained |
| webqsp | in-domain | zrckpn | +2.1 (+0.7, +3.4) R above empty | +1.3 (-0.2, +2.9) level | -18.2 (-20.0, -16.5) R-trained below N-trained | +36.8 (+34.6, +39.1) S-trained above N-trained |
