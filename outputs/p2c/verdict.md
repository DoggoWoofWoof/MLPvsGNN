# P2c part 1: LLM-free triples against LLM-built graphs on B1

## musique (1000 questions, 11656 passages)

Agreement (recall / precision, %):

| reference | graph | phrases | pairs | P edges |
| --- | --- | --- | --- | --- |
| H4o | ours | 73.3 / 43.5 | 14.7 / 11.5 | 60.6 / 23.5 |
| H4o | NER | 40.3 / 72.4 | 0.0 / - | 42.3 / 52.4 |
| H4o | ours-noREL | 73.3 / 43.5 | 0.0 / - | 60.6 / 23.5 |
| H70 | ours | 77.7 / 39.7 | 12.2 / 10.0 | 62.9 / 25.3 |
| H70 | NER | 48.9 / 75.4 | 0.0 / - | 47.9 / 61.7 |
| H70 | ours-noREL | 77.7 / 39.7 | 0.0 / - | 62.9 / 25.3 |
| SR | ours | 86.6 / 26.6 | 8.1 / 7.0 | 77.6 / 20.8 |
| SR | NER | 72.2 / 66.5 | 0.0 / - | 68.5 / 58.6 |
| SR | ours-noREL | 86.6 / 26.6 | 0.0 / - | 77.6 / 20.8 |

R@5 (%) in each walk:

| walk | NER | ours | ours-noREL | H4o | H4o-noREL | H70 | H70-noREL | SR | SR-noREL | dense |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WH | 36.2 | 47.9 | 44.1 | 44.4 | 38.7 | 44.6 | 40.2 | 45.8 | 44.8 | 56.7 |

| WP-T5 | today | today+ours | today+H4o | today+H70 | today+SR |
| --- | --- | --- | --- | --- | --- |
| R@5 | 56.4 | 56.4 | 56.4 | 56.4 | 56.4 |

| WP-RW | today | today+ours | today+H4o | today+H70 | today+SR |
| --- | --- | --- | --- | --- | --- |
| R@5 | 28.8 | 37.2 | 33.6 | 33.7 | 32.5 |

The share of each reference's gain that ours recovers:

| walk | reference | reference's gain [95%] | our gain [95%] | ours − reference | share [95%] | call |
| --- | --- | --- | --- | --- | --- | --- |
| WP-T5 | H4o | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 | - | NO_REFERENCE_GAIN |
| WP-T5 | H70 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 | - | NO_REFERENCE_GAIN |
| WP-T5 | SR | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 | - | NO_REFERENCE_GAIN |
| WP-RW | H4o | +4.77 [+3.69, +5.82] | +8.37 [+7.07, +9.66] | +3.60 | 1.76 [1.49, 2.14] | MATCHED |
| WP-RW | H70 | +4.83 [+3.81, +5.92] | +8.37 [+7.07, +9.66] | +3.53 | 1.73 [1.49, 2.05] | MATCHED |
| WP-RW | SR | +3.65 [+2.67, +4.65] | +8.37 [+7.07, +9.66] | +4.72 | 2.29 [1.89, 2.92] | MATCHED |
| WH | H4o | +8.15 [+6.20, +9.97] | +11.62 [+9.86, +13.38] | +3.48 | 1.43 [1.22, 1.72] | MATCHED |
| WH | H70 | +8.34 [+6.44, +10.29] | +11.62 [+9.86, +13.38] | +3.28 | 1.39 [1.18, 1.70] | MATCHED |
| WH | SR | +9.62 [+7.90, +11.37] | +11.62 [+9.86, +13.38] | +2.01 | 1.21 [1.05, 1.42] | MATCHED |

WH questions with a matched phrase: NER 0.998, ours 1.000, ours-noREL 1.000, H4o 1.000, H4o-noREL 1.000, H70 1.000, H70-noREL 1.000, SR 0.992, SR-noREL 0.992

## 2wiki (1000 questions, 6119 passages)

Agreement (recall / precision, %):

| reference | graph | phrases | pairs | P edges |
| --- | --- | --- | --- | --- |
| SR | ours | 83.8 / 26.2 | 7.2 / 6.2 | 78.0 / 18.5 |
| SR | NER | 69.3 / 59.9 | 0.0 / - | 67.6 / 49.5 |
| SR | ours-noREL | 83.8 / 26.2 | 0.0 / - | 78.0 / 18.5 |

R@5 (%) in each walk:

| walk | NER | ours | ours-noREL | SR | SR-noREL | dense |
| --- | --- | --- | --- | --- | --- | --- |
| WH | 54.5 | 67.2 | 66.6 | 71.1 | 76.6 | 70.8 |

| WP-T5 | today | today+ours | today+SR |
| --- | --- | --- | --- |
| R@5 | 71.2 | 71.2 | 71.2 |

| WP-RW | today | today+ours | today+SR |
| --- | --- | --- | --- |
| R@5 | 47.4 | 51.2 | 48.9 |

The share of each reference's gain that ours recovers:

| walk | reference | reference's gain [95%] | our gain [95%] | ours − reference | share [95%] | call |
| --- | --- | --- | --- | --- | --- | --- |
| WP-T5 | SR | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 | - | NO_REFERENCE_GAIN |
| WP-RW | SR | +1.50 [+0.57, +2.43] | +3.82 [+2.55, +5.05] | +2.32 | 2.55 [1.58, 5.86] | MATCHED |
| WH | SR | +16.55 [+14.72, +18.40] | +12.70 [+10.98, +14.50] | -3.85 | 0.77 [0.69, 0.84] | PARTIAL |

WH questions with a matched phrase: NER 0.997, ours 1.000, ours-noREL 1.000, SR 0.988, SR-noREL 0.988

