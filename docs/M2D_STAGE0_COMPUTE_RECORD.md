# M2D Stage-0 compute record

Filed at `ab2092f1d70f`, before any job was submitted. CPU only; this record prices no accelerator and authorises none.

## What is authorised

Four CPU-only jobs, one per declared Stage-0 cell, each producing the complementarity, fixed-fusion and rescue diagnostics for that cell.

Nothing is trained, no weight is fitted and no test split is opened. Every number the jobs produce comes from four rankings over an already-frozen candidate pool.

## Container

| cores | memory | GPU | timeout | rate |
|---:|---:|---|---:|---:|
| 4 | 16 GiB | none | 3600 s | $0.3172/h |

The probe runs two forward passes per query over a few hundred candidates and does no backward pass. An accelerator would spend money on transfer, not on arithmetic.

## Measured inputs

- Structural columns: **9**, derived from the recorded parameter counts in the M2B baseline table, which pin it exactly given a Linear(n, 32) + Linear(32, 1) scorer.
- Scoring kernel at a pool of 400: **20.20 ms/query** for both rungs together (S3 12.14, S4 8.06; 14.75 to 28.03 across samples). Measured on this host, now, on the forward pass the probe performs, over 51 timed calls per rung. The priced figure is the median, inflated by the stated container safety factor for the walltime verdict; the fastest and slowest samples are recorded beside it. A host estimate, not a container measurement.
- Largest bracketed mean pool: 372.5 candidates, so the kernel measurement over-prices every cell rather than fitting the cheapest.

## Workload

| cell | panel | mean pool | predicted seconds |
|---|---:|---:|---:|
| squad_clean/R1 | 20,850 | 318.6 | 421 |
| musique_clean/R1 | 3,190 | 331.9 | 64 |
| hotpotqa_clean/R1 | 15,656 | 372.5 | 316 |
| metaqa/R1 | 31,310 | 372.5 | 632 |

71,006 queries across 4 jobs. Every cell runs its whole fit portion. musique_clean holds 3,190 queries and is the smallest; capping it to save seconds would spend the precision the +0.25pp gate needs, on the one cell that can least afford it.

## Prediction

- Measured work: **1434 s** total.
- Largest single job: **632 s** (metaqa/R1), against a 3600 s timeout.
- Feasible within the timeout at 1.5x safety: **True** — fits in one window
- At the slowest kernel sample this host produced, the largest job would be **878 s** and the whole workload $0.44. That figure prices the machine's other work, so it caps the bill rather than settling the verdict.
- Expected spend at 40% utilisation: **$0.32**.
- Hard ceiling: **$1.00**, which covers the slowest-sample figure as well.

## Abort criteria

- any job exceeding its predicted walltime above by more than the stated container safety factor
- a container acquiring a GPU, which this record does not price
- a cell whose panel size differs from the figure above, which would mean the sealed master is not the one M2B fitted
- cumulative spend reaching the hard ceiling

## What this record does not authorise

- any Stage-1 fit
- seeds 1 and 2
- the full 14-cell screen
- any GPU hour
