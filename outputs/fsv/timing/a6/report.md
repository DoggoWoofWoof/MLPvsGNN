# FSV timing, Family 1, A6 forms (M5)

Family 1 is zrckpn (the MLP) against zg6r (the GNN), every input kept, each at the stage forms of amendment A2 (`docs/FSV_FAST_VERSIONS.md` section 10) with amendment A4's forms of stage (a) and of the GNN's layers (section 12) and amendment A6's forms of stage (b) and of the encoder (section 13). This file is M5, the timing of those forms; the passes of A4's forms (M3) and of A2's own forms (M2) stand as the record of them. Times are ms per question at batch 1 on the host's CPU, cold unless labelled warm; each cell is p50 / p95 over the kept questions. Stage (a), the shared pool, is charged in full to both models: model-only = (b) + (c), end to end = (a) + (b) + (c). A faster exact form found later is an amendment and a re-time of these numbers. Family 2 (the fast versions) is not in this file.

Ratios are GNN / MLP of the p50s, paired by question, with the 2.5th and 97.5th percentiles over 2,000 resamples of the questions with replacement (`numpy.random.default_rng(0)`).

## BLAS 2 (the record)

### Cold, per stage (ms, p50 / p95)

| dataset | kept | pool rows p50 | (a) | MLP (b) | MLP (c) | MLP model-only | MLP end to end | GNN (b) | GNN (c) | GNN model-only | GNN end to end |
|---|---|---|---|---|---|---|---|---|---|---|---|
| musique | 200 / 200 | 4814 | 133.9 / 197.3 | 65.6 / 151.2 | 13.2 / 23.3 | 78.3 / 172.8 | 218.0 / 361.3 | 65.0 / 135.6 | 50.9 / 90.2 | 115.0 / 225.8 | 255.3 / 407.3 |
| hotpotqa | 200 / 200 | 697 | 163.4 / 215.7 | 11.8 / 20.3 | 2.2 / 3.4 | 14.4 / 23.0 | 176.6 / 236.0 | 8.6 / 18.1 | 5.9 / 8.7 | 15.2 / 26.0 | 177.3 / 234.3 |
| 2wiki | 200 / 200 | 1074 | 176.8 / 213.8 | 17.0 / 25.5 | 3.2 / 4.4 | 20.0 / 29.9 | 197.1 / 237.8 | 13.2 / 22.1 | 9.0 / 12.9 | 23.1 / 32.0 | 201.7 / 243.7 |

### Ratio GNN / MLP, and stage (a)'s share of each end-to-end time

| dataset | (b) | (c) | model-only | end to end | (a) p50 ms | (a) share of MLP end to end | (a) share of GNN end to end |
|---|---|---|---|---|---|---|---|
| musique | 0.99 [0.92, 1.03] | 3.86 [3.72, 4.04] | 1.47 [1.39, 1.55] | 1.17 [1.14, 1.19] | 133.9 | 61% | 52% |
| hotpotqa | 0.72 [0.61, 0.87] | 2.64 [2.58, 2.74] | 1.06 [0.96, 1.22] | 1.00 [0.99, 1.02] | 163.4 | 93% | 92% |
| 2wiki | 0.78 [0.73, 0.88] | 2.81 [2.70, 2.87] | 1.15 [1.06, 1.24] | 1.02 [1.01, 1.04] | 176.8 | 90% | 88% |

### Parts of (a) and (b) (p50 ms)

| dataset | (a) pool | base | links | rule: walks | orders | KT | other | MLP compile | rows | links | NEXT | inputs | GNN compile | rows | links | inputs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| musique | 0.3 | 0.1 | 0.1 | 62.9 | 64.1 | 5.5 | 0.1 | 35.5 | 20.3 | 2.2 | 4.5 | 2.6 | 37.1 | 20.1 | 2.3 | 4.8 |
| hotpotqa | 0.1 | 0.1 | 0.1 | 100.1 | 54.0 | 3.8 | 0.1 | 5.7 | 1.6 | 0.1 | 2.6 | 0.6 | 5.6 | 1.6 | 0.1 | 0.8 |
| 2wiki | 0.1 | 0.1 | 0.1 | 107.4 | 62.7 | 4.2 | 0.1 | 9.4 | 2.5 | 0.1 | 3.3 | 1.0 | 9.2 | 2.5 | 0.1 | 1.2 |

### Warm (the forward alone, per question; labelled warm)

| dataset | MLP b1 | GNN b1 | ratio b1 | MLP b16 | GNN b16 | ratio b16 |
|---|---|---|---|---|---|---|
| musique | 12.5 / 17.9 | 55.4 / 89.6 | 4.42 [4.22, 4.56] | 13.0 / 18.7 | 205.5 / 314.8 | 15.80 [15.61, 16.01] |
| hotpotqa | 2.1 / 3.3 | 6.0 / 9.3 | 2.92 [2.81, 3.05] | 1.9 / 2.3 | 12.7 / 14.4 | 6.86 [6.64, 7.08] |
| 2wiki | 2.9 / 3.9 | 8.2 / 11.3 | 2.82 [2.73, 2.92] | 2.9 / 4.5 | 17.2 / 20.7 | 5.85 [5.77, 5.93] |

### Checks, pin and the host's load (1 s samples before and after the timed pass)

| dataset | questions failing a check | pool failing | score checks failing MLP / GNN | max score diff MLP / GNN | NEXT values differing | code conflicts | pinned | load before: pinned LPs / all mean / max | load after: pinned LPs / all mean / max |
|---|---|---|---|---|---|---|---|---|---|
| musique | 0 of 200 | 0 | 0 / 0 | 1.8e-05 / 9.5e-06 | 6 | 21 | True | 56% / 35% / 100% | 25% / 26% / 100% |
| hotpotqa | 0 of 200 | 0 | 0 / 0 | 3.3e-06 / 5.4e-06 | 1 | 1 | True | 33% / 21% / 79% | 15% / 19% / 67% |
| 2wiki | 0 of 200 | 0 | 0 / 0 | 3.8e-06 / 7.6e-06 | 4 | 7 | True | 25% / 25% / 83% | 23% / 25% / 86% |

### Noise, beside (p50 ms; deciding nothing)

| dataset | (a): first / second half | MLP model-only: first / second half | GNN model-only: first / second half | MLP led first / second | GNN led first / second | p95 / p50 of (a), MLP, GNN |
|---|---|---|---|---|---|---|
| musique | 137 / 132 | 83 / 76 | 120 / 110 | 83 / 76 | 108 / 121 | 1.47, 2.21, 1.96 |
| hotpotqa | 163 / 164 | 13 / 16 | 15 / 17 | 17 / 11 | 20 / 12 | 1.32, 1.60, 1.71 |
| 2wiki | 172 / 185 | 20 / 20 | 22 / 24 | 24 / 16 | 26 / 20 | 1.21, 1.50, 1.39 |

## BLAS 1 (beside the record: every question timed, checks counted)

### Cold, per stage (ms, p50 / p95)

| dataset | kept | pool rows p50 | (a) | MLP (b) | MLP (c) | MLP model-only | MLP end to end | GNN (b) | GNN (c) | GNN model-only | GNN end to end |
|---|---|---|---|---|---|---|---|---|---|---|---|
| musique | 200 / 200 | 4814 | 141.4 / 194.5 | 68.7 / 139.7 | 13.2 / 19.8 | 82.3 / 157.6 | 225.1 / 342.6 | 64.1 / 131.2 | 50.6 / 81.6 | 112.6 / 206.5 | 258.8 / 387.5 |
| hotpotqa | 200 / 200 | 697 | 163.3 / 212.5 | 10.0 / 15.5 | 2.3 / 3.8 | 12.4 / 19.5 | 176.2 / 225.3 | 7.3 / 12.4 | 6.3 / 9.4 | 13.4 / 21.3 | 177.4 / 227.0 |
| 2wiki | 200 / 200 | 1074 | 172.5 / 206.0 | 14.9 / 20.7 | 3.3 / 4.3 | 18.0 / 24.8 | 190.9 / 226.4 | 11.5 / 16.7 | 8.9 / 12.1 | 20.6 / 29.2 | 193.4 / 230.8 |

### Ratio GNN / MLP, and stage (a)'s share of each end-to-end time

| dataset | (b) | (c) | model-only | end to end | (a) p50 ms | (a) share of MLP end to end | (a) share of GNN end to end |
|---|---|---|---|---|---|---|---|
| musique | 0.93 [0.91, 1.00] | 3.84 [3.67, 4.00] | 1.37 [1.34, 1.47] | 1.15 [1.14, 1.18] | 141.4 | 63% | 55% |
| hotpotqa | 0.73 [0.68, 0.77] | 2.78 [2.69, 2.85] | 1.08 [1.03, 1.16] | 1.01 [1.00, 1.02] | 163.3 | 93% | 92% |
| 2wiki | 0.78 [0.74, 0.82] | 2.74 [2.63, 2.82] | 1.14 [1.09, 1.19] | 1.01 [1.00, 1.02] | 172.5 | 90% | 89% |

### Parts of (a) and (b) (p50 ms)

| dataset | (a) pool | base | links | rule: walks | orders | KT | other | MLP compile | rows | links | NEXT | inputs | GNN compile | rows | links | inputs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| musique | 0.3 | 0.1 | 0.1 | 61.4 | 69.8 | 5.4 | 0.1 | 38.1 | 20.4 | 2.2 | 4.5 | 2.9 | 36.6 | 20.0 | 2.3 | 5.4 |
| hotpotqa | 0.1 | 0.1 | 0.1 | 102.4 | 51.1 | 4.0 | 0.1 | 4.6 | 1.7 | 0.1 | 2.7 | 0.6 | 4.6 | 1.7 | 0.1 | 0.8 |
| 2wiki | 0.1 | 0.1 | 0.1 | 108.1 | 58.3 | 4.1 | 0.1 | 7.9 | 2.5 | 0.1 | 3.1 | 0.9 | 7.8 | 2.5 | 0.1 | 1.1 |

### Warm (the forward alone, per question; labelled warm)

| dataset | MLP b1 | GNN b1 | ratio b1 | MLP b16 | GNN b16 | ratio b16 |
|---|---|---|---|---|---|---|
| musique | 12.3 / 17.1 | 53.4 / 87.8 | 4.33 [4.16, 4.50] | 13.6 / 18.5 | 201.5 / 313.8 | 14.82 [14.61, 15.04] |
| hotpotqa | 2.1 / 3.3 | 6.0 / 9.0 | 2.82 [2.69, 2.90] | 1.7 / 2.1 | 11.6 / 12.4 | 6.62 [6.47, 6.77] |
| 2wiki | 2.9 / 4.2 | 8.1 / 10.7 | 2.75 [2.67, 2.84] | 2.8 / 3.2 | 16.6 / 19.5 | 5.93 [5.84, 6.02] |

### Checks, pin and the host's load (1 s samples before and after the timed pass)

| dataset | questions failing a check | pool failing | score checks failing MLP / GNN | max score diff MLP / GNN | NEXT values differing | code conflicts | pinned | load before: pinned LPs / all mean / max | load after: pinned LPs / all mean / max |
|---|---|---|---|---|---|---|---|---|---|
| musique | 23 of 200 | 0 | 16 / 22 | 0.0021 / 0.0013 | 6 | 21 | True | 61% / 18% / 78% | 37% / 18% / 80% |
| hotpotqa | 4 of 200 | 0 | 3 / 2 | 0.0013 / 0.0013 | 1 | 1 | True | 11% / 22% / 73% | 34% / 21% / 81% |
| 2wiki | 3 of 200 | 0 | 0 / 3 | 4e-05 / 0.00043 | 4 | 7 | True | 35% / 23% / 86% | 27% / 16% / 66% |

### Noise, beside (p50 ms; deciding nothing)

| dataset | (a): first / second half | MLP model-only: first / second half | GNN model-only: first / second half | MLP led first / second | GNN led first / second | p95 / p50 of (a), MLP, GNN |
|---|---|---|---|---|---|---|
| musique | 140 / 142 | 84 / 79 | 117 / 109 | 83 / 81 | 113 / 113 | 1.38, 1.91, 1.83 |
| hotpotqa | 166 / 159 | 13 / 12 | 14 / 13 | 13 / 11 | 14 / 13 | 1.30, 1.57, 1.59 |
| 2wiki | 177 / 170 | 20 / 17 | 22 / 19 | 20 / 17 | 21 / 20 | 1.19, 1.37, 1.42 |

## Caveats

- The host is shared: other jobs run on it while a timed job does, so every time carries that noise. The process is pinned to two logical processors (its own affinity, EcoQoS opted out, above-normal priority), the models alternate in leading each question, and the load of every logical processor is sampled for 1 s before and after the pass; the other jobs are not pinned and may share those processors.
- The first-stage retrieval (the question's encoding, the dense and SPLADE lists) is not timed; PKO's prepared lists are read before timing as index-time data.
- Stage (a) is PKO's rule's own calls on one question at batch 1 on one thread, with the walks on P4b's records and only the orders the pool and the KT columns read (amendment A4); the GNN's layer step is A4's; stage (b)'s links of both models, the MLP's NEXT columns and the encoder's segment z-scores of both models are A6's (eager torch, one thread); OpenBLAS runs on 2 threads in the record and on 1 in the beside pass.
- Warm times are the forward alone on inputs built beforehand; b16 is the time of 12 batches of the first 192 questions over 16, so it has no values for the last 8 questions.
- Three of the development jobs of the b16 layer step (amendment A7, section 14) ran on logical processors 8 and 10 while a record pass was timed on 2 and 4: the hotpotqa job (22:54 to 22:59 IST, 4 Nov) beside hotpotqa's BLAS-2 pass, the musique job (22:59 to 23:04) beside hotpotqa's BLAS-1 pass, and the 2wiki job (23:14 to 23:20) beside musique's BLAS-2 pass, whose whole timed region it covered; the other three passes ran with no such job. A noise read, made after the passes and decided by no rule, found no slowdown beside those jobs: stage (a), the same code in M3 and M5, has a p50 of 133.9 ms in musique's BLAS-2 pass against 141.4 in its BLAS-1 pass (no job beside it) and 142.7 in M3; the drift between the halves of that pass in model-only p50 (MLP 83 to 76 ms, GNN 120 to 110) is the drift of the BLAS-1 pass (84 to 79, 117 to 109), common to both models; and the ratios of the two halves agree (cold forward 3.91 and 3.86, end to end 1.160 and 1.175). No pass was rerun. The b16 columns here are the module's layer step on the batches of 16; amendment A7's re-time replaces them where its checks pass. This note was added after the generator ran.
