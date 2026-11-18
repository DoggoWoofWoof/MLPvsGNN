# ENC E1b part 2: each encoder's real pools under PKO's rule (report)

Written by `outputs/mp_unified/zenc1b2.py report` under docs/ENC_ENCODER_COMPARISON.md section 11. Shares are % of s1eval golds; gaps are points against the search view (the frozen encoder through the same swaps), paired 95% bootstrap.

## musique

- GV (record view = PKO's filed parts): PASS 
- GS (search view against the filed pools): PASS {"equal_pool_share": 0.997931, "gold_share_diff_pts": 0.0, "verdict": "PASS"}
- filed: pool 97.767 (B 76.0618); search: pool 97.767 (B 76.0618), 5263.62 rows

| arm | pool share | pool gap | B gap | won back | rows (x search) |
| --- | ---: | --- | --- | ---: | ---: |
| qwen3e06 | 97.1893 | -0.58 [-0.86, -0.30] BELOW | -3.23 [-3.89, -2.58] | +2.65 | 1.0289 |
| mpnet | 97.2205 | -0.55 [-0.85, -0.24] BELOW | -2.98 [-3.62, -2.32] | +2.43 | 0.9982 |
| mpnetqa | 96.9706 | -0.80 [-1.11, -0.50] BELOW | -3.83 [-4.45, -3.20] | +3.03 | 0.995 |

## metaqa

- GV (record view = PKO's filed parts): PASS 
- GS (search view against the filed pools): PASS {"equal_pool_share": 0.999796, "gold_share_diff_pts": 0.0, "verdict": "PASS"}
- filed: pool 98.8376 (B 1.4223); search: pool 98.8376 (B 1.4223), 4373.68 rows

| arm | pool share | pool gap | B gap | won back | rows (x search) |
| --- | ---: | --- | --- | ---: | ---: |
| qwen3e06 | 98.5997 | -0.24 [-0.40, -0.09] BELOW | -0.56 [-0.66, -0.45] | +0.32 | 1.0003 |
| mpnet | 97.6478 | -1.19 [-1.45, -0.93] BELOW | -0.34 [-0.46, -0.22] | -0.85 | 0.988 |
| mpnetqa | 97.0411 | -1.80 [-2.10, -1.49] BELOW | -0.75 [-0.85, -0.65] | -1.05 | 0.9806 |

## The calls (section 11.5; SQuAD's gap is part 1's, its pool being B)

- **qwen3e06: CLOSE** {"musique": [-0.58, -0.86], "metaqa": [-0.24, -0.4], "squad": [-0.8844, -1.0781]}
- **mpnet: LOSES** {"musique": [-0.55, -0.85], "metaqa": [-1.19, -1.45], "squad": [-1.0949, -1.3139]}
- **mpnetqa: LOSES** {"musique": [-0.8, -1.11], "metaqa": [-1.8, -2.1], "squad": [-0.9602, -1.1623]}
