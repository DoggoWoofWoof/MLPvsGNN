# FSV timing, Family 1, A4 forms (M3)

Family 1 is zrckpn (the MLP) against zg6r (the GNN), every input kept, each at the stage forms of amendment A2 (`docs/FSV_FAST_VERSIONS.md` section 10) with amendment A4's forms of stage (a) and of the GNN's layers (section 12). This file is M3, the timing of those forms; the passes of A2's own forms (M2) stand as the record of them. Times are ms per question at batch 1 on the host's CPU, cold unless labelled warm; each cell is p50 / p95 over the kept questions. Stage (a), the shared pool, is charged in full to both models: model-only = (b) + (c), end to end = (a) + (b) + (c). A faster exact form found later is an amendment and a re-time of these numbers. Family 2 (the fast versions) is not in this file.

Ratios are GNN / MLP of the p50s, paired by question, with the 2.5th and 97.5th percentiles over 2,000 resamples of the questions with replacement (`numpy.random.default_rng(0)`).

## BLAS 2 (the record)

### Cold, per stage (ms, p50 / p95)

| dataset | kept | pool rows p50 | (a) | MLP (b) | MLP (c) | MLP model-only | MLP end to end | GNN (b) | GNN (c) | GNN model-only | GNN end to end |
|---|---|---|---|---|---|---|---|---|---|---|---|
| musique | 200 / 200 | 4814 | 142.7 / 200.5 | 83.2 / 173.3 | 27.0 / 38.2 | 110.8 / 210.3 | 260.3 / 410.9 | 77.2 / 164.0 | 69.2 / 104.2 | 142.4 / 260.3 | 296.9 / 461.2 |
| hotpotqa | 200 / 200 | 697 | 154.6 / 218.3 | 12.0 / 20.6 | 5.5 / 7.7 | 17.6 / 26.9 | 171.9 / 241.9 | 8.5 / 16.4 | 8.8 / 12.7 | 17.8 / 27.8 | 173.2 / 238.3 |
| 2wiki | 200 / 200 | 1074 | 178.8 / 215.8 | 18.7 / 28.0 | 7.5 / 10.2 | 26.1 / 38.3 | 205.1 / 247.5 | 15.0 / 23.0 | 12.7 / 17.3 | 27.7 / 38.5 | 208.8 / 246.8 |

### Ratio GNN / MLP, and stage (a)'s share of each end-to-end time

| dataset | (b) | (c) | model-only | end to end | (a) p50 ms | (a) share of MLP end to end | (a) share of GNN end to end |
|---|---|---|---|---|---|---|---|
| musique | 0.93 [0.89, 0.99] | 2.56 [2.46, 2.66] | 1.29 [1.26, 1.37] | 1.14 [1.11, 1.16] | 142.7 | 55% | 48% |
| hotpotqa | 0.71 [0.64, 0.83] | 1.60 [1.56, 1.64] | 1.02 [0.93, 1.08] | 1.01 [0.99, 1.02] | 154.6 | 90% | 89% |
| 2wiki | 0.80 [0.71, 0.87] | 1.69 [1.64, 1.73] | 1.06 [0.99, 1.15] | 1.02 [1.00, 1.04] | 178.8 | 87% | 86% |

### Parts of (a) and (b) (p50 ms)

| dataset | (a) pool | base | links | rule: walks | orders | KT | other | MLP compile | rows | links | NEXT | inputs | GNN compile | rows | links | inputs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| musique | 0.3 | 0.1 | 0.1 | 63.2 | 71.1 | 5.5 | 0.1 | 38.2 | 21.1 | 12.9 | 7.4 | 3.0 | 37.2 | 21.0 | 13.3 | 5.4 |
| hotpotqa | 0.1 | 0.1 | 0.1 | 97.9 | 50.8 | 3.7 | 0.1 | 5.3 | 1.7 | 0.4 | 2.8 | 0.6 | 5.4 | 1.6 | 0.4 | 0.7 |
| 2wiki | 0.1 | 0.1 | 0.1 | 110.0 | 61.8 | 4.1 | 0.1 | 9.9 | 2.6 | 0.7 | 3.6 | 0.9 | 9.7 | 2.5 | 0.7 | 1.1 |

### Warm (the forward alone, per question; labelled warm)

| dataset | MLP b1 | GNN b1 | ratio b1 | MLP b16 | GNN b16 | ratio b16 |
|---|---|---|---|---|---|---|
| musique | 26.4 / 37.1 | 72.4 / 112.4 | 2.74 [2.62, 2.86] | 30.7 / 37.4 | 237.2 / 376.8 | 7.73 [7.58, 7.89] |
| hotpotqa | 6.8 / 14.0 | 9.0 / 17.7 | 1.32 [1.23, 1.40] | 3.4 / 3.7 | 13.4 / 18.4 | 3.93 [3.86, 4.01] |
| 2wiki | 7.7 / 10.6 | 14.2 / 19.0 | 1.84 [1.78, 1.90] | 5.2 / 5.9 | 20.4 / 23.2 | 3.94 [3.83, 4.05] |

### Checks, pin and the host's load (1 s samples before and after the timed pass)

| dataset | questions failing a check | pool failing | score checks failing MLP / GNN | max score diff MLP / GNN | NEXT values differing | code conflicts | pinned | load before: pinned LPs / all mean / max | load after: pinned LPs / all mean / max |
|---|---|---|---|---|---|---|---|---|---|
| musique | 0 of 200 | 0 | 0 / 0 | 1.8e-05 / 9.5e-06 | 6 | 21 | True | 36% / 24% / 88% | 38% / 34% / 89% |
| hotpotqa | 0 of 200 | 0 | 0 / 0 | 3.3e-06 / 5.4e-06 | 1 | 1 | True | 27% / 13% / 67% | 42% / 19% / 80% |
| 2wiki | 0 of 200 | 0 | 0 / 0 | 3.8e-06 / 7.6e-06 | 4 | 7 | True | 31% / 22% / 64% | 27% / 22% / 78% |

### Noise, beside (p50 ms; deciding nothing)

| dataset | (a): first / second half | MLP model-only: first / second half | GNN model-only: first / second half | MLP led first / second | GNN led first / second | p95 / p50 of (a), MLP, GNN |
|---|---|---|---|---|---|---|
| musique | 139 / 146 | 111 / 109 | 145 / 140 | 112 / 108 | 141 / 146 | 1.40, 1.90, 1.83 |
| hotpotqa | 149 / 162 | 17 / 19 | 17 / 20 | 20 / 16 | 21 / 15 | 1.41, 1.53, 1.56 |
| 2wiki | 178 / 180 | 27 / 26 | 28 / 27 | 30 / 22 | 31 / 25 | 1.21, 1.46, 1.39 |

## BLAS 1 (beside the record: every question timed, checks counted)

### Cold, per stage (ms, p50 / p95)

| dataset | kept | pool rows p50 | (a) | MLP (b) | MLP (c) | MLP model-only | MLP end to end | GNN (b) | GNN (c) | GNN model-only | GNN end to end |
|---|---|---|---|---|---|---|---|---|---|---|---|
| musique | 200 / 200 | 4814 | 132.5 / 183.5 | 75.5 / 168.9 | 25.3 / 36.1 | 103.5 / 204.8 | 237.7 / 369.2 | 70.2 / 155.9 | 61.1 / 102.8 | 133.1 / 259.8 | 270.0 / 430.3 |
| hotpotqa | 200 / 200 | 697 | 181.4 / 240.0 | 11.8 / 18.8 | 6.2 / 9.1 | 18.4 / 28.6 | 196.4 / 261.1 | 8.3 / 15.0 | 9.9 / 15.1 | 18.0 / 30.0 | 198.4 / 260.2 |
| 2wiki | 200 / 200 | 1074 | 181.3 / 216.9 | 16.5 / 23.2 | 7.6 / 10.1 | 24.2 / 32.1 | 205.2 / 244.1 | 12.7 / 18.6 | 12.8 / 17.4 | 25.8 / 35.5 | 208.0 / 246.2 |

### Ratio GNN / MLP, and stage (a)'s share of each end-to-end time

| dataset | (b) | (c) | model-only | end to end | (a) p50 ms | (a) share of MLP end to end | (a) share of GNN end to end |
|---|---|---|---|---|---|---|---|
| musique | 0.93 [0.88, 0.97] | 2.41 [2.34, 2.49] | 1.29 [1.25, 1.36] | 1.14 [1.11, 1.15] | 132.5 | 56% | 49% |
| hotpotqa | 0.70 [0.66, 0.74] | 1.59 [1.52, 1.66] | 0.98 [0.94, 1.03] | 1.01 [0.99, 1.01] | 181.4 | 92% | 91% |
| 2wiki | 0.77 [0.74, 0.80] | 1.68 [1.65, 1.76] | 1.06 [1.03, 1.09] | 1.01 [1.01, 1.02] | 181.3 | 88% | 87% |

### Parts of (a) and (b) (p50 ms)

| dataset | (a) pool | base | links | rule: walks | orders | KT | other | MLP compile | rows | links | NEXT | inputs | GNN compile | rows | links | inputs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| musique | 0.3 | 0.1 | 0.1 | 59.0 | 63.4 | 5.3 | 0.1 | 35.1 | 19.6 | 12.0 | 6.8 | 2.6 | 35.0 | 19.7 | 11.8 | 4.9 |
| hotpotqa | 0.1 | 0.1 | 0.1 | 110.9 | 58.7 | 4.1 | 0.1 | 5.3 | 1.8 | 0.5 | 3.2 | 0.7 | 5.2 | 1.8 | 0.5 | 0.8 |
| 2wiki | 0.1 | 0.1 | 0.1 | 115.1 | 60.2 | 4.2 | 0.1 | 8.4 | 2.6 | 0.7 | 3.6 | 0.9 | 8.2 | 2.6 | 0.7 | 1.1 |

### Warm (the forward alone, per question; labelled warm)

| dataset | MLP b1 | GNN b1 | ratio b1 | MLP b16 | GNN b16 | ratio b16 |
|---|---|---|---|---|---|---|
| musique | 26.9 / 35.6 | 71.1 / 109.3 | 2.65 [2.55, 2.73] | 25.4 / 37.1 | 221.1 / 331.3 | 8.70 [8.55, 8.86] |
| hotpotqa | 5.6 / 7.9 | 9.1 / 12.5 | 1.61 [1.56, 1.67] | 3.6 / 4.0 | 13.9 / 15.4 | 3.82 [3.73, 3.92] |
| 2wiki | 8.0 / 10.7 | 12.7 / 16.2 | 1.58 [1.55, 1.64] | 5.4 / 6.0 | 19.3 / 22.4 | 3.60 [3.53, 3.67] |

### Checks, pin and the host's load (1 s samples before and after the timed pass)

| dataset | questions failing a check | pool failing | score checks failing MLP / GNN | max score diff MLP / GNN | NEXT values differing | code conflicts | pinned | load before: pinned LPs / all mean / max | load after: pinned LPs / all mean / max |
|---|---|---|---|---|---|---|---|---|---|
| musique | 23 of 200 | 0 | 16 / 22 | 0.0021 / 0.0013 | 6 | 21 | True | 38% / 21% / 85% | 38% / 40% / 86% |
| hotpotqa | 4 of 200 | 0 | 3 / 2 | 0.0013 / 0.0013 | 1 | 1 | True | 60% / 44% / 89% | 41% / 32% / 83% |
| 2wiki | 3 of 200 | 0 | 0 / 3 | 4e-05 / 0.00043 | 4 | 7 | True | 18% / 20% / 81% | 35% / 21% / 85% |

### Noise, beside (p50 ms; deciding nothing)

| dataset | (a): first / second half | MLP model-only: first / second half | GNN model-only: first / second half | MLP led first / second | GNN led first / second | p95 / p50 of (a), MLP, GNN |
|---|---|---|---|---|---|---|
| musique | 133 / 131 | 104 / 99 | 134 / 124 | 103 / 103 | 130 / 135 | 1.39, 1.98, 1.95 |
| hotpotqa | 182 / 179 | 18 / 19 | 17 / 18 | 19 / 18 | 19 / 18 | 1.32, 1.55, 1.66 |
| 2wiki | 180 / 182 | 24 / 24 | 26 / 25 | 26 / 23 | 26 / 26 | 1.20, 1.32, 1.38 |

## Caveats

- The host is shared: other jobs run on it while a timed job does, so every time carries that noise. The process is pinned to two logical processors (its own affinity, EcoQoS opted out, above-normal priority), the models alternate in leading each question, and the load of every logical processor is sampled for 1 s before and after the pass; the other jobs are not pinned and may share those processors.
- The first-stage retrieval (the question's encoding, the dense and SPLADE lists) is not timed; PKO's prepared lists are read before timing as index-time data.
- Stage (a) is PKO's rule's own calls on one question at batch 1 on one thread, with the walks on P4b's records and only the orders the pool and the KT columns read (amendment A4); the GNN's layer step is A4's and the MLP's forward the module's (eager torch, one thread); OpenBLAS runs on 2 threads in the record and on 1 in the beside pass.
- Warm times are the forward alone on inputs built beforehand; b16 is the time of 12 batches of the first 192 questions over 16, so it has no values for the last 8 questions.
- Musique's BLAS 1 pass was run twice. The first overlapped in time with a duplicate launch of the same job (cancelled), which was loading and setting up on the same two logical processors; it was repeated alone (--overwrite; the raw head says overwrote: true) and only the repeat is reported here. The first pass's checks were the same (23 of 200 failing, the BLAS 1 count of M2's pass); it is kept aside and used for no number. This note was added after the generator ran.
