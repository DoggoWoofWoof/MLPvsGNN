# ENC E1b part 1: each encoder's base pool B

Declared in docs/ENC_ENCODER_COMPARISON.md section 10. Shares are % of golds in B; gaps are points against E0's frozen arm (95%, 2,000 resamples).

## musique (2,417 questions, 6,404 golds, equal_rrf_budget_200)

Frozen: E0 B 76.06; filed B 76.06; filed pool 97.77; X (expansion) +21.71. G3 equal share 1.0.

| arm | gold share | all golds | mean B | gap (pts, 95%) | call | gate |
|---|---|---|---|---|---|---|
| qwen3e06 | 72.83 | 45.80 | 200.0 | -3.23 [-3.89, -2.58] | BELOW | NEAR |
| mpnet | 73.08 | 46.59 | 200.0 | -2.98 [-3.62, -2.32] | BELOW | NEAR |
| mpnetqa | 72.24 | 44.68 | 200.0 | -3.83 [-4.45, -3.20] | BELOW | NEAR |

## squad (11,873 questions, 11,873 golds, equal_rrf_budget_50)

Frozen: E0 B 97.98; filed B 97.98; filed pool 97.98; X (expansion) +0.00. G3 equal share 1.0.

| arm | gold share | all golds | mean B | gap (pts, 95%) | call | gate |
|---|---|---|---|---|---|---|
| qwen3e06 | 97.09 | 97.09 | 50.0 | -0.88 [-1.08, -0.70] | BELOW | FAIL |
| mpnet | 96.88 | 96.88 | 50.0 | -1.09 [-1.31, -0.88] | BELOW | FAIL |
| mpnetqa | 97.02 | 97.02 | 50.0 | -0.96 [-1.16, -0.76] | BELOW | FAIL |

## metaqa (9,785 questions, 72,697 golds, equal_rrf_budget_50)

Frozen: E0 B 1.42; filed B 1.42; filed pool 98.84; X (expansion) +97.42. G3 equal share 1.0.

| arm | gold share | all golds | mean B | gap (pts, 95%) | call | gate |
|---|---|---|---|---|---|---|
| qwen3e06 | 0.86 | 0.91 | 50.0 | -0.56 [-0.66, -0.45] | BELOW | PASS |
| mpnet | 1.08 | 2.38 | 50.0 | -0.34 [-0.46, -0.22] | BELOW | PASS |
| mpnetqa | 0.67 | 1.36 | 50.1 | -0.75 [-0.85, -0.65] | BELOW | PASS |

## The calls

- **qwen3e06: HOLD**
- **mpnet: HOLD**
- **mpnetqa: HOLD**
