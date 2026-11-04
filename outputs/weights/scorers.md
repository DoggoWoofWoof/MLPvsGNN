# Weight count of the next-hop scorer behind zrckpn's NEXT columns

Read on 5 Nov 2026 from the saved fold files of the scorer (`outputs/weights/count_scorers.py`, read-only). The two models' own counts are in weights.md (same folder), unchanged.

| part | weights |
|---|---:|
| net (14 -> 32 -> 32 -> 1, with biases) | 1,569 |
| V (3 x 1536 x 32) | 147,456 |
| U (3 x 1536 x 32) | 147,456 |
| **one scorer** | **296,481** |

The feature mean and spread stored with each fold file (14 numbers each) are constants, not weights.

## Which scorers a fit reads

- Each split has 5 fold fits of this one shape (10 files in all, one set of shapes). WQ-musique's fit record is 8456ed7419da, WQ-hotpotqa's is 4026adb96bb6; every fold file matches the sha256 its record names.
- A question reads exactly one scorer, never an average of the five (docs/LTM_NEXT_HOP_MLP.md section 2). A training question's columns come from the fold fit that never saw its component. An evaluation question's columns come from the fold fit its index draws, uniform over the 5. Over a whole fit, all 5 fold fits of its split are used.
- The six zrckpn fits are the same in this respect. Seeds 0, 1 and 2 on WQ-musique read WQ-musique's five; the three `-hp` fits read WQ-hotpotqa's five. Each fit's read record names its split's fit record by sha256, and it matches the files read here. Each fit's NEXT block is 7 columns (next_1, next_2, next_3, next_4, next_5, next_max, next_n).

## Sizes side by side

| system | weights |
|---|---:|
| zrckpn alone (weights.md) | 226,950 |
| zrckpn and the one scorer a question reads | 523,431 |
| zrckpn and all 5 fold fits of one split | 1,709,355 |
| zg6r (weights.md) | 404,955 |

weights.md's "1.78 times" counts the MLP alone. With the one scorer its NEXT columns need, the MLP's side holds 1.29 times the GNN's weights (523,431 against 404,955). The GNN of record reads no NEXT block (its saved input widths are rank, dense_cos, topo_STRUCT, depth_STRUCT, SEMB, SEED, WALK, DISTS and WALKF, zrckpn's list without NEXT), so it has no scorer to add. NEXT widens zrckpn's first layer by 15 inputs (docs/LTM_NEXT_HOP_MLP.md section 9): its first layer is 324 wide in all six fits against 309 in the six zg6r fits; those first-layer weights are already inside the 226,950. Neither count includes the frozen text encoder.
