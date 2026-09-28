# UMLP-D0.2: joint separability of the golds the twin misses (2wiki)

Analysis only (configs/umlp_d02_joint_separability.yaml). Question: can the frozen compiled representation jointly distinguish an additional gold recovered by the GNN from the non-golds occupying the twin's top-5 slots? Population: the D0.1 sidecar (2wiki V2_GATE, integrity-checked); held half not read; M3B GAT not loaded; no retrieval model touched.

**Reading: D02_R1_LINEARLY_PRESENT.** Bar: out-of-fold within-query pair-AUC lower 95% bound > 0.6 on all three twin seeds.

## Primary task: G_miss_mp vs N_disp (out-of-fold)

| twin seed | queries | golds | pairs | A retained linear | B retained MLP | C raw-164 MLP |
|---|---:|---:|---:|---|---|---|
| s0 | 865 | 911 | 2,920 | 0.808 [0.790, 0.827] | 0.799 [0.780, 0.818] | 0.815 [0.797, 0.833] |
| s1 | 808 | 829 | 2,618 | 0.695 [0.670, 0.718] | 0.723 [0.698, 0.748] | 0.745 [0.723, 0.767] |
| s2 | 923 | 964 | 3,065 | 0.825 [0.805, 0.841] | 0.827 [0.808, 0.845] | 0.842 [0.824, 0.859] |

Pair accuracy (unweighted): s0 A 0.817 / B 0.807 / C 0.819; s1 A 0.697 / B 0.732 / C 0.748; s2 A 0.820 / B 0.821 / C 0.838.

Clears the bar on all three seeds: {'A': True, 'B': True, 'C': True}.

## Reference orderings on the same pairs (descriptive)

| twin seed | fixed base | twin residual | twin score | u_gnn_v2_ef |
|---|---|---|---|---|
| s0 | 0.397 [0.376, 0.418] | 0.242 [0.222, 0.262] | 0.000 [0.000, 0.000] | 0.836 [0.820, 0.850] |
| s1 | 0.402 [0.381, 0.424] | 0.155 [0.138, 0.173] | 0.000 [0.000, 0.000] | 0.821 [0.803, 0.836] |
| s2 | 0.462 [0.442, 0.481] | 0.174 [0.157, 0.191] | 0.000 [0.000, 0.000] | 0.837 [0.823, 0.851] |

## Secondary (descriptive, no verdict): all G_miss vs N_disp

| twin seed | queries | A | B | C | fixed base | twin residual | GNN |
|---|---:|---|---|---|---|---|---|
| s0 | 1,293 | 0.792 [0.776, 0.807] | 0.790 [0.774, 0.806] | 0.797 [0.781, 0.814] | 0.398 [0.381, 0.414] | 0.205 [0.189, 0.220] | 0.627 [0.605, 0.647] |
| s1 | 1,249 | 0.706 [0.688, 0.724] | 0.722 [0.705, 0.740] | 0.729 [0.711, 0.747] | 0.405 [0.389, 0.423] | 0.125 [0.112, 0.139] | 0.609 [0.588, 0.629] |
| s2 | 1,355 | 0.827 [0.812, 0.842] | 0.825 [0.812, 0.839] | 0.840 [0.826, 0.853] | 0.457 [0.441, 0.474] | 0.150 [0.137, 0.163] | 0.649 [0.630, 0.668] |

## Selection note (read before the reading)

The pairs are conditioned on the twin's own ranking: every negative is a candidate the twin placed at ranks 2-5 and every positive a gold it placed below 5. On such a population, features correlated with what the twin already rewards mark the negatives, so part of any probe's separation is a reversal of the twin's preference, not new evidence. The trivial reversal -- the inverted fixed base -- already reaches s0 0.603, s1 0.598, s2 0.538 on the primary pairs; probe A exceeds it by s0 +0.205, s1 +0.097, s2 +0.286.

The probes are fitted on this conditional population only. A clear bar shows the retained columns jointly order these pairs out of fold; it does not show that one full-pool scorer can lift these golds without demoting the golds the twin already ranks well. That is a retrieval question, and a v2.1 declaration would have to test it on retrieval metrics.


## Protocol

- Cross-fitting: 5 folds, fold = sha256(query_id) mod 5; all pairs of a query in one fold; queries per fold on the primary task (s0): [180, 158, 180, 180, 167].
- Pairs: d = x_gold - x_neg entered as (d, 1) and (-d, 0); each query weighs 1 in total. Normalisation fitted on training folds only.
- Probe form: per-candidate scorer g(x), pair logit g(x_gold) - g(x_neg) (for A exactly logistic regression on d without intercept).
- A: LogisticRegression l2 C=1 no intercept lbfgs max_iter 5000; 130 parameters.
- B, C: Linear(k,64)-GELU-Linear(64,1), AdamW lr 0.001 wd 0.0001, 500 full-batch steps, seed 0; B 8,449 / C 10,689 parameters.
- Inputs: A/B 130 (129 surviving + fixed base; sha256 `a01b8113c7b6f708`), C 165 (164 raw + fixed base; sha256 `0009d88cceb12c48`). No GNN score, twin score, residual, rank, arm or dataset id is an input.

## Incidents

- ruling's 129 / 164 verified against feature_screen.json before filing (surviving 129 == twin read set, columns 164, dropped 35)

## Artefacts

- d01_sidecar: `outputs/umlp_d01/candidates_2wiki_gate.npz` sha256 `4b18d3d5fcff005e`
- d01_record: `outputs/umlp_d01/record.json` sha256 `62b0d6bea89d970e`
- query_ids: `outputs/universal_v2/eval/2wiki_query_ids.json` sha256 `d439a33c22f39186`
- feature_screen: `outputs/universal_v2/feature_screen.json` sha256 `c58b317e9ae0f345`
- record: `outputs/umlp_d02/record.json` sha256 `6f2c2d5dff05b6ca`
