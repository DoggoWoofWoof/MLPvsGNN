# FSV timing, Family 1, A2 forms

Family 1 is zrckpn (the MLP) against zg6r (the GNN), every input kept, each at the stage forms of amendment A2 (`docs/FSV_FAST_VERSIONS.md` section 10). Times are ms per question at batch 1 on the host's CPU, cold unless labelled warm; each cell is p50 / p95 over the kept questions. Stage (a), the shared pool, is charged in full to both models: model-only = (b) + (c), end to end = (a) + (b) + (c). A faster exact form found later is an amendment and a re-time of these numbers. Family 2 (the fast versions) is not in this file.

Ratios are GNN / MLP of the p50s, paired by question, with the 2.5th and 97.5th percentiles over 2,000 resamples of the questions with replacement (`numpy.random.default_rng(0)`).

## BLAS 2 (the record)

### Cold, per stage (ms, p50 / p95)

| dataset | kept | pool rows p50 | (a) | MLP (b) | MLP (c) | MLP model-only | MLP end to end | GNN (b) | GNN (c) | GNN model-only | GNN end to end |
|---|---|---|---|---|---|---|---|---|---|---|---|
| musique | 200 / 200 | 4814 | 419.6 / 592.0 | 76.8 / 169.8 | 25.9 / 34.0 | 102.9 / 201.5 | 527.2 / 775.0 | 73.5 / 152.3 | 206.8 / 394.2 | 279.5 / 551.2 | 700.5 / 1111.5 |
| hotpotqa | 200 / 200 | 697 | 1173.5 / 1478.8 | 11.2 / 18.6 | 6.2 / 8.6 | 17.4 / 26.3 | 1187.5 / 1495.7 | 8.1 / 13.0 | 13.2 / 22.8 | 21.5 / 35.2 | 1190.4 / 1501.3 |
| 2wiki | 200 / 200 | 1074 | 1353.1 / 1718.3 | 16.7 / 25.3 | 7.9 / 10.5 | 25.0 / 35.2 | 1377.5 / 1742.0 | 13.1 / 19.8 | 18.9 / 34.1 | 32.0 / 51.2 | 1383.1 / 1755.1 |

### Ratio GNN / MLP, and stage (a)'s share of each end-to-end time

| dataset | (b) | (c) | model-only | end to end | (a) p50 ms | (a) share of MLP end to end | (a) share of GNN end to end |
|---|---|---|---|---|---|---|---|
| musique | 0.96 [0.91, 1.00] | 8.00 [7.13, 8.50] | 2.72 [2.54, 2.82] | 1.33 [1.28, 1.37] | 419.6 | 80% | 60% |
| hotpotqa | 0.72 [0.69, 0.77] | 2.14 [2.04, 2.24] | 1.24 [1.19, 1.29] | 1.00 [1.00, 1.01] | 1173.5 | 99% | 99% |
| 2wiki | 0.78 [0.74, 0.81] | 2.40 [2.28, 2.51] | 1.28 [1.23, 1.37] | 1.00 [1.00, 1.01] | 1353.1 | 98% | 98% |

### Parts of (a) and (b) (p50 ms)

| dataset | (a) pool | base | links | rule: walks | orders | KT | other | MLP compile | rows | links | NEXT | inputs | GNN compile | rows | links | inputs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| musique | 0.3 | 0.1 | 0.1 | 242.7 | 151.6 | 13.5 | 2.5 | 35.8 | 19.9 | 12.0 | 7.0 | 2.8 | 36.2 | 20.2 | 12.6 | 5.1 |
| hotpotqa | 0.1 | 0.1 | 0.1 | 1046.2 | 110.3 | 12.5 | 2.8 | 4.9 | 1.8 | 0.5 | 3.0 | 0.6 | 5.0 | 1.8 | 0.5 | 0.8 |
| 2wiki | 0.1 | 0.1 | 0.1 | 1202.9 | 132.8 | 12.5 | 3.2 | 8.5 | 2.6 | 0.7 | 3.5 | 0.9 | 8.5 | 2.6 | 0.7 | 1.1 |

### Warm (the forward alone, per question; labelled warm)

| dataset | MLP b1 | GNN b1 | ratio b1 | MLP b16 | GNN b16 | ratio b16 |
|---|---|---|---|---|---|---|
| musique | 26.9 / 36.4 | 214.9 / 447.6 | 7.98 [7.30, 8.56] | 26.7 / 35.7 | 229.0 / 374.8 | 8.59 [8.27, 8.90] |
| hotpotqa | 6.0 / 8.1 | 15.4 / 25.0 | 2.56 [2.45, 2.66] | 3.8 / 5.0 | 16.3 / 22.0 | 4.33 [4.18, 4.50] |
| 2wiki | 7.5 / 9.8 | 19.8 / 32.2 | 2.65 [2.52, 2.81] | 5.5 / 6.4 | 21.2 / 26.3 | 3.85 [3.79, 3.90] |

### Checks, pin and the host's load (1 s samples before and after the timed pass)

| dataset | questions failing a check | pool failing | score checks failing MLP / GNN | max score diff MLP / GNN | NEXT values differing | code conflicts | pinned | load before: pinned LPs / all mean / max | load after: pinned LPs / all mean / max |
|---|---|---|---|---|---|---|---|---|---|
| musique | 0 of 200 | 0 | 0 / 0 | 1.8e-05 / 9.5e-06 | 6 | 21 | True | 29% / 33% / 91% | 50% / 49% / 86% |
| hotpotqa | 0 of 200 | 0 | 0 / 0 | 3.3e-06 / 5.4e-06 | 1 | 1 | True | 16% / 33% / 85% | 30% / 29% / 88% |
| 2wiki | 0 of 200 | 0 | 0 / 0 | 3.8e-06 / 7.6e-06 | 4 | 7 | True | 37% / 29% / 91% | 28% / 29% / 78% |

### Noise, beside (p50 ms; deciding nothing)

| dataset | (a): first / second half | MLP model-only: first / second half | GNN model-only: first / second half | MLP led first / second | GNN led first / second | p95 / p50 of (a), MLP, GNN |
|---|---|---|---|---|---|---|
| musique | 415 / 424 | 106 / 102 | 295 / 261 | 106 / 102 | 273 / 285 | 1.41, 1.96, 1.97 |
| hotpotqa | 1149 / 1243 | 17 / 17 | 21 / 22 | 18 / 16 | 22 / 21 | 1.26, 1.51, 1.64 |
| 2wiki | 1478 / 1308 | 25 / 25 | 33 / 31 | 27 / 24 | 33 / 31 | 1.27, 1.41, 1.60 |

## BLAS 1 (beside the record: every question timed, checks counted)

### Cold, per stage (ms, p50 / p95)

| dataset | kept | pool rows p50 | (a) | MLP (b) | MLP (c) | MLP model-only | MLP end to end | GNN (b) | GNN (c) | GNN model-only | GNN end to end |
|---|---|---|---|---|---|---|---|---|---|---|---|
| musique | 200 / 200 | 4814 | 412.8 / 597.7 | 77.4 / 152.0 | 25.3 / 33.1 | 103.0 / 182.6 | 523.7 / 778.2 | 74.1 / 144.4 | 192.6 / 379.7 | 267.1 / 523.5 | 699.3 / 1121.2 |
| hotpotqa | 200 / 200 | 697 | 1097.0 / 1389.1 | 11.1 / 19.2 | 6.2 / 8.2 | 17.3 / 27.0 | 1115.8 / 1408.7 | 8.2 / 13.4 | 13.2 / 22.2 | 21.1 / 34.1 | 1122.4 / 1415.0 |
| 2wiki | 200 / 200 | 1074 | 1802.6 / 2265.0 | 23.6 / 35.2 | 10.0 / 14.1 | 33.8 / 49.3 | 1837.9 / 2296.2 | 18.3 / 30.1 | 26.0 / 52.4 | 45.0 / 76.9 | 1856.3 / 2321.4 |

### Ratio GNN / MLP, and stage (a)'s share of each end-to-end time

| dataset | (b) | (c) | model-only | end to end | (a) p50 ms | (a) share of MLP end to end | (a) share of GNN end to end |
|---|---|---|---|---|---|---|---|
| musique | 0.96 [0.90, 1.00] | 7.62 [7.04, 8.39] | 2.59 [2.45, 2.77] | 1.34 [1.29, 1.37] | 412.8 | 79% | 59% |
| hotpotqa | 0.74 [0.69, 0.77] | 2.12 [2.04, 2.18] | 1.22 [1.17, 1.26] | 1.01 [1.00, 1.01] | 1097.0 | 98% | 98% |
| 2wiki | 0.78 [0.74, 0.83] | 2.59 [2.46, 2.70] | 1.33 [1.28, 1.40] | 1.01 [1.00, 1.02] | 1802.6 | 98% | 97% |

### Parts of (a) and (b) (p50 ms)

| dataset | (a) pool | base | links | rule: walks | orders | KT | other | MLP compile | rows | links | NEXT | inputs | GNN compile | rows | links | inputs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| musique | 0.3 | 0.1 | 0.1 | 236.2 | 164.2 | 12.9 | 4.0 | 36.5 | 19.5 | 12.2 | 7.0 | 2.7 | 36.8 | 19.9 | 12.2 | 5.1 |
| hotpotqa | 0.1 | 0.1 | 0.1 | 974.4 | 105.0 | 12.5 | 2.4 | 5.1 | 1.7 | 0.5 | 3.0 | 0.6 | 5.0 | 1.8 | 0.5 | 0.8 |
| 2wiki | 0.1 | 0.1 | 0.1 | 1598.8 | 183.7 | 14.5 | 4.0 | 12.6 | 3.3 | 0.8 | 5.0 | 1.4 | 12.7 | 3.3 | 0.8 | 1.6 |

### Warm (the forward alone, per question; labelled warm)

| dataset | MLP b1 | GNN b1 | ratio b1 | MLP b16 | GNN b16 | ratio b16 |
|---|---|---|---|---|---|---|
| musique | 26.8 / 35.5 | 212.5 / 448.1 | 7.93 [7.42, 8.66] | 26.0 / 33.6 | 225.0 / 350.4 | 8.65 [8.53, 8.77] |
| hotpotqa | 6.4 / 8.9 | 13.5 / 23.6 | 2.09 [2.01, 2.21] | 3.8 / 4.5 | 13.7 / 15.2 | 3.64 [3.53, 3.75] |
| 2wiki | 8.9 / 11.3 | 22.7 / 39.3 | 2.56 [2.44, 2.78] | 6.5 / 7.7 | 28.0 / 33.7 | 4.31 [4.22, 4.40] |

### Checks, pin and the host's load (1 s samples before and after the timed pass)

| dataset | questions failing a check | pool failing | score checks failing MLP / GNN | max score diff MLP / GNN | NEXT values differing | code conflicts | pinned | load before: pinned LPs / all mean / max | load after: pinned LPs / all mean / max |
|---|---|---|---|---|---|---|---|---|---|
| musique | 23 of 200 | 0 | 16 / 22 | 0.0021 / 0.0013 | 6 | 21 | True | 31% / 27% / 77% | 28% / 22% / 85% |
| hotpotqa | 4 of 200 | 0 | 3 / 2 | 0.0013 / 0.0013 | 1 | 1 | True | 43% / 19% / 76% | 31% / 37% / 91% |
| 2wiki | 3 of 200 | 0 | 0 / 3 | 4e-05 / 0.00043 | 4 | 7 | True | 17% / 61% / 100% | 35% / 51% / 98% |

### Noise, beside (p50 ms; deciding nothing)

| dataset | (a): first / second half | MLP model-only: first / second half | GNN model-only: first / second half | MLP led first / second | GNN led first / second | p95 / p50 of (a), MLP, GNN |
|---|---|---|---|---|---|---|
| musique | 405 / 421 | 105 / 101 | 286 / 249 | 102 / 104 | 267 / 267 | 1.45, 1.77, 1.96 |
| hotpotqa | 1063 / 1176 | 17 / 18 | 21 / 21 | 19 / 17 | 22 / 21 | 1.27, 1.55, 1.62 |
| 2wiki | 1752 / 1931 | 35 / 32 | 45 / 45 | 36 / 31 | 45 / 45 | 1.26, 1.46, 1.71 |

## Caveats

- The host is shared: other jobs run on it while a timed job does, so every time carries that noise. The process is pinned to two logical processors (its own affinity, EcoQoS opted out, above-normal priority), the models alternate in leading each question, and the load of every logical processor is sampled for 1 s before and after the pass; the other jobs are not pinned and may share those processors.
- The first-stage retrieval (the question's encoding, the dense and SPLADE lists) is not timed; PKO's prepared lists are read before timing as index-time data.
- Stage (a) is PKO's own rule on one question at batch 1 on one thread; no fused forward exists for either model (eager torch, one thread); OpenBLAS runs on 2 threads in the record and on 1 in the beside pass.
- Warm times are the forward alone on inputs built beforehand; b16 is the time of 12 batches of the first 192 questions over 16, so it has no values for the last 8 questions.
