# FSV timing, Family 1, the warm batches of 16 (A7)

Family 1 is zrckpn (the MLP) against zg6r (the GNN), every input kept, at amendment A4's, A6's and A7's forms (`docs/FSV_FAST_VERSIONS.md` sections 12 to 14). Times are ms per question: a batch of 16's forward time over 16, on the host's CPU, p50 / p95 over the batches and rounds. The GNN is timed with the module's layer step (what M5's warm batches run) and with A7's batched step, the MLP once, in the same job. Ratios are GNN / MLP of the p50s, paired by batch and round, with the 2.5th and 97.5th percentiles over 2,000 resamples of the batch times with replacement (`numpy.random.default_rng(0)`).

## BLAS 2 (the record)

| dataset | MLP | GNN, module's step | GNN, A7's step | GNN speed-up | ratio, module's step | ratio, A7's step | same bytes (batches) | failing vs reads (MLP / GNN module / GNN A7) |
|---|---|---|---|---|---|---|---|---|
| musique | 11.68 / 18.50 | 181.33 / 277.61 | 52.83 / 70.70 | 3.43x | 15.52 [14.85, 16.14] | 4.52 [4.34, 4.66] | 12 / 12 | 0 / 0 / 0 of 192 |
| hotpotqa | 1.93 / 2.69 | 10.80 / 12.52 | 5.99 / 7.48 | 1.80x | 5.60 [5.34, 5.84] | 3.10 [2.97, 3.22] | 12 / 12 | 0 / 0 / 0 of 192 |
| 2wiki | 2.80 / 5.37 | 16.32 / 24.14 | 9.05 / 15.20 | 1.80x | 5.83 [5.45, 6.55] | 3.23 [3.03, 3.53] | 12 / 12 | 0 / 0 / 0 of 192 |

## BLAS 1 (beside the record)

| dataset | MLP | GNN, module's step | GNN, A7's step | GNN speed-up | ratio, module's step | ratio, A7's step | same bytes (batches) | failing vs reads (MLP / GNN module / GNN A7) |
|---|---|---|---|---|---|---|---|---|
| musique | 11.77 / 15.42 | 180.95 / 275.90 | 52.40 / 74.61 | 3.45x | 15.37 [14.85, 16.30] | 4.45 [4.36, 4.62] | 12 / 12 | 0 / 0 / 0 of 192 |
| hotpotqa | 1.92 / 2.48 | 10.75 / 12.36 | 6.07 / 7.48 | 1.77x | 5.59 [5.40, 5.80] | 3.16 [3.03, 3.31] | 12 / 12 | 0 / 0 / 0 of 192 |
| 2wiki | 2.62 / 3.27 | 14.97 / 19.49 | 8.48 / 11.14 | 1.77x | 5.72 [5.48, 5.87] | 3.24 [3.15, 3.31] | 12 / 12 | 0 / 0 / 0 of 192 |

## Caveats

- The host is shared: other jobs run on it while a timed job does, so every time carries that noise. The process is pinned to two logical processors (its own affinity, EcoQoS opted out, above-normal priority), the load of every logical processor is sampled for 1 s before and after each pass (in report.json), and the other jobs are not pinned and may share those processors. No other timed job of this track ran beside these passes: the smoke and the three runs went one at a time, after M5's last pass.
- Each cell is the forward alone, per question (a batch of 16's forward time over 16): the 12 batches of the first 192 questions over five counted rounds (60 batch times) after a round that is not counted. The three forwards of a batch (the MLP's, the GNN's with the module's step, the GNN's with A7's) are timed in an order that rotates, the features are built outside the clock, and both BLAS counts run in one process on inputs built once at BLAS 2 (OpenBLAS on 2 threads in the record, 1 in the pass beside).
- A7's batched step gave the module's step's scores byte for byte in all 12 batches of every dataset at both BLAS counts; every variant's scores stood within 1e-4 of the fit's filed read, with the same top five, on all 192 questions (largest differences at BLAS 2: musique MLP 1.8e-05, GNN 7.2e-06; hotpotqa 2.4e-06, 6.0e-06; 2wiki 1.9e-06, 5.7e-06); every raw head says stop False, and the counter of the batched step held (432 layer steps a job, none the module's). The ratio GNN / MLP falls with A7's step because the GNN's largest stage at that batch shrank and the MLP's did not; the cold forward and end to end are not touched. For the warm batch of 16 the number of record is this report's, on all three datasets; M5's cells (one forward a batch, the module's step; report ed3cdd1) stay in M5's report.
