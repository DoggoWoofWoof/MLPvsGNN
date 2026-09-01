# Validation-only phase screen

Status: **complete only when every registered cell below is present**.

Positive values mean the seed-aware GNN exceeds QLS-MLP on validation R@5. No test metric is read by this analysis.

## 2wiki_clean

- `degree_rewire` — gaps [0.00: +1.75, 0.10: +1.18, 0.25: +0.02, 0.50: +0.07, 1.00: +0.14]; selected [0.00, 1.00]
- `random_add` — gaps [0.00: +1.75, 0.10: +1.05, 0.25: +1.08, 0.50: +0.53, 1.00: +0.60]; selected [0.00, 1.00]
- `hub_injection` — gaps [0.00: +1.75, 0.10: +1.58, 0.25: +1.15, 0.50: +1.12, 1.00: -0.60]; selected [0.00, 0.50, 1.00]
- `feature_mask` — gaps [0.00: +1.75, 0.25: +1.02, 0.50: +0.23, 0.75: +0.53, 1.00: -5.18]; selected [0.00, 0.75, 1.00]

## musique_clean

- `degree_rewire` — gaps [0.00: +0.24, 0.10: -0.19, 0.25: +0.06, 0.50: -0.04, 1.00: +0.01]; selected [0.00, 0.10, 0.25, 0.50, 1.00]
- `random_add` — gaps [0.00: +0.24, 0.10: +0.49, 0.25: -0.01, 0.50: +0.34, 1.00: +0.72]; selected [0.00, 0.10, 0.25, 0.50, 1.00]
- `hub_injection` — gaps [0.00: +0.24, 0.10: +0.18, 0.25: -0.18, 0.50: +0.09, 1.00: -1.42]; selected [0.00, 0.10, 0.25, 0.50, 1.00]
- `feature_mask` — gaps [0.00: +0.24, 0.25: -0.11, 0.50: +0.52, 0.75: +1.32, 1.00: -5.78]; selected [0.00, 0.25, 0.50, 0.75, 1.00]

## webqsp

- `degree_rewire` — gaps [0.00: -0.49, 0.10: +0.64, 0.25: +1.03, 0.50: -1.52, 1.00: -1.61]; selected [0.00, 0.10, 0.25, 0.50, 1.00]
- `random_add` — gaps [0.00: -0.49, 0.10: -0.28, 0.25: -0.05, 0.50: -0.64, 1.00: -0.76]; selected [0.00, 1.00]
- `hub_injection` — gaps [0.00: -0.49, 0.10: -0.11, 0.25: -0.46, 0.50: -1.03, 1.00: -2.39]; selected [0.00, 1.00]
- `feature_mask` — gaps [0.00: -0.49, 0.25: -1.23, 0.50: -2.96, 0.75: -5.16, 1.00: -12.29]; selected [0.00, 1.00]

## hotpotqa_clean

- `degree_rewire` — gaps [0.00: +0.24, 0.10: +0.16, 0.25: -0.48, 0.50: -0.48, 1.00: -0.32]; selected [0.00, 0.10, 0.25, 1.00]
- `random_add` — gaps [0.00: +0.24, 0.10: +0.06, 0.25: +0.25, 0.50: +0.21, 1.00: +0.30]; selected [0.00, 1.00]
- `hub_injection` — gaps [0.00: +0.24, 0.10: +0.23, 0.25: -0.43, 0.50: -0.73, 1.00: -0.94]; selected [0.00, 0.10, 0.25, 1.00]
- `feature_mask` — gaps [0.00: +0.24, 0.25: +0.31, 0.50: -0.13, 0.75: -0.82, 1.00: -4.79]; selected [0.00, 0.25, 0.50, 1.00]

## squad_clean

- `degree_rewire` — gaps [0.00: -0.12, 0.10: +0.13, 0.25: +0.02, 0.50: -0.13, 1.00: -0.29]; selected [0.00, 0.10, 0.25, 0.50, 1.00]
- `random_add` — gaps [0.00: -0.12, 0.10: +0.08, 0.25: +0.01, 0.50: +0.24, 1.00: -0.03]; selected [0.00, 0.10, 0.50, 1.00]
- `hub_injection` — gaps [0.00: -0.12, 0.10: +0.09, 0.25: +0.08, 0.50: -0.22, 1.00: -0.69]; selected [0.00, 0.10, 0.25, 0.50, 1.00]
- `feature_mask` — gaps [0.00: -0.12, 0.25: +0.02, 0.50: +0.11, 0.75: +0.18, 1.00: -6.39]; selected [0.00, 0.25, 0.75, 1.00]

## metaqa

- `degree_rewire` — gaps [0.00: -0.06, 0.10: -0.04, 0.25: +0.03, 0.50: -0.04, 1.00: -0.41]; selected [0.00, 0.10, 0.25, 0.50, 1.00]
- `random_add` — gaps [0.00: -0.06, 0.10: -0.03, 0.25: +0.02, 0.50: +0.03, 1.00: +0.09]; selected [0.00, 0.10, 0.25, 1.00]
- `hub_injection` — gaps [0.00: -0.06, 0.10: +0.28, 0.25: +0.34, 0.50: -0.13, 1.00: -0.69]; selected [0.00, 0.10, 0.25, 0.50, 1.00]
- `feature_mask` — gaps [0.00: -0.06, 0.25: -0.03, 0.50: -0.14, 0.75: -0.22, 1.00: -6.06]; selected [0.00, 1.00]

## Frozen next boundary

The union of selected rates is written to the generated phase-confirmation configuration. That file must be reviewed, committed, and tagged before any selected test cell runs. Predictor fitting remains prohibited until the five-seed confirmation establishes reproducible help, neutral, and harm regions.
