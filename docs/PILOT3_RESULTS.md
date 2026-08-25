# Three-dataset protocol pilot: results and next gate

Date: 2026-08-25

> **Status: `NOT_PAPER_VALID_PILOT`. None of the numbers in this document may
> appear as paper evidence.** WebQSP, 2Wiki, and MuSiQue currently come from
> incomplete/test-only L2 caches that were temporarily re-split. The purpose of
> this run is to validate the comparison contract, execution system, statistics,
> and perturbation behavior before importing canonical datasets and splits.

## Decision

The protocol is ready for a narrow continuation of the topology sweep, not for
a claim that MLPs beat GNNs. On the clean pilot, a one-layer GCN has higher R@5
than the parameter-matched MLP on all three datasets. Degree-preserving rewiring
of 25% of edge positions reduces that advantage on all three; MuSiQue moves to
an effectively neutral interval, while no dataset crosses into a reliable MLP
win. The immediate next experiment is therefore the remaining registered
rewiring rates, followed by random-edge addition if the curve remains coherent.
Hub injection and feature degradation stay gated behind that result. The LODO
predictor remains prohibited until clear help, neutral, and harm regions exist.

## Frozen baseline and run scope

- protocol commit and annotated tag: `paper-protocol-v0`
  (`7817f7d082ade588863d705bb80f62c3a76e3de7`);
- execution: Modal A10G, one dataset per parallel function;
- models: one-layer GCN, parameter-matched MLP, and analytically
  compute-matched MLP;
- training: seed 0, two epochs, hidden size 32 for GCN/parameter MLP, dropout
  0.2, learning rate 0.001;
- metrics: R@1, R@5, R@10, R@20, R@50, MRR, full coverage, conditional recall,
  candidate ceiling, and latency;
- uncertainty: paired query bootstrap, shown only as a pilot diagnostic;
- first perturbation: degree-preserving rewiring at nominal rate 0.25 with seed
  31415.

The temporary splits are 111/24/24 for WebQSP, 1050/225/225 for 2Wiki, and
1396/299/300 for MuSiQue. One seed, two epochs, and these temporary splits are
not an adequate statistical study.

## Fair-comparison audit

Each result file stores SHA-256 hashes for the candidate/split assignment,
labels, raw frozen expert evidence, shared model inputs, and GNN topology. The
runner enforces the following contract:

- candidates, relevance labels, listwise loss, optimizer/training loop, and
  seed schedule are paired;
- raw frozen features and model input tensors are identical;
- the MLP receives `edge_index=None`;
- the GCN's only extra information is the induced candidate topology;
- the parameter arm has exactly 2,465 trainable parameters for each model;
- the compute arm matches deterministic analytical sparse MAC estimates within
  1.1% on every dataset.

Across clean and rewired runs, all non-topology hashes remain identical, the
topology hash changes, and every non-latency MLP metric is bit-identical. This
is the key intervention sanity check: the topology perturbation reaches only
the GCN.

The compute-matched widths are 35 for WebQSP, 33 for 2Wiki, and 34 for MuSiQue,
against GCN width 32. The relative analytical-MAC gaps are 0.70%, 1.08%, and
0.50%, respectively. Measured GPU wall-clock gaps remain roughly 43--47%
because sparse kernels do not map FLOPs to elapsed time like dense kernels.
Wall time is therefore a diagnostic, not the matching criterion; the canonical
study must add an equal-accelerator-time sensitivity analysis.

## Clean pilot

All retrieval values below are fractions. GCN-minus-MLP deltas use the
parameter-matched MLP.

| Dataset | Model | R@1 | R@5 | R@20 | MRR | ms/query |
|---|---|---:|---:|---:|---:|---:|
| WebQSP | parameter MLP | 0.2101 | 0.3719 | 0.5642 | 0.4681 | 0.80 |
| WebQSP | compute MLP | 0.2517 | 0.3636 | 0.5225 | 0.4845 | 0.84 |
| WebQSP | GCN | 0.2581 | 0.4133 | 0.6964 | 0.4976 | 2.05 |
| 2Wiki | parameter MLP | 0.3978 | 0.7189 | 0.7878 | 0.9401 | 0.73 |
| 2Wiki | compute MLP | 0.4078 | 0.7167 | 0.7878 | 0.9546 | 0.80 |
| 2Wiki | GCN | 0.3967 | 0.8044 | 0.8822 | 0.9366 | 1.92 |
| MuSiQue | parameter MLP | 0.3833 | 0.7375 | 0.8831 | 0.8975 | 0.66 |
| MuSiQue | compute MLP | 0.3789 | 0.7353 | 0.8836 | 0.8947 | 0.66 |
| MuSiQue | GCN | 0.3894 | 0.7492 | 0.8894 | 0.9059 | 1.67 |

| Dataset | GCN - parameter MLP R@5 | Pilot 95% paired CI |
|---|---:|---:|
| WebQSP | +0.0414 | [-0.0417, +0.1490] |
| 2Wiki | +0.0856 | [+0.0600, +0.1111] |
| MuSiQue | +0.0117 | [+0.0011, +0.0228] |

These intervals condition on one trained seed and temporary data. They quantify
query variation only and must not be read as model-training uncertainty.

## Controlled topology intervention

The rewiring implementation preserves the in/out degree multiset and edge
count, rejects multiset-equivalent swaps, handles duplicate directed edges, and
uses an edge position at most once. The audited achieved rates are:

| Dataset | Changed edge positions | Achieved fraction |
|---|---:|---:|
| WebQSP | 508,430 / 2,033,748 | 0.249997 |
| 2Wiki | 1,726,780 / 6,906,992 | 0.250005 |
| MuSiQue | 3,440,816 / 13,763,344 | 0.249999 |

| Dataset | Clean GCN R@5 | Rewired GCN R@5 | Clean gap | Rewired gap | Gap change |
|---|---:|---:|---:|---:|---:|
| WebQSP | 0.4133 | 0.4025 | +0.0414 | +0.0306 | -0.0108 |
| 2Wiki | 0.8044 | 0.7767 | +0.0856 | +0.0578 | -0.0278 |
| MuSiQue | 0.7492 | 0.7408 | +0.0117 | +0.0033 | -0.0083 |

The rewired R@5 paired intervals are [-0.0352, +0.1285] for WebQSP,
[+0.0356, +0.0800] for 2Wiki, and [-0.0100, +0.0167] for MuSiQue. MuSiQue has
entered the neutral region under the pre-registered one-point practical margin,
but the study has not yet observed a robust negative crossover.

R@20 does not always move with R@5: in WebQSP, for example, the rewired GCN's
R@20 rises. Together with the earlier R@1/R@5 smoke reversal, this confirms that
the paper must report a metric-by-regime surface rather than reduce the result
to one winner label.

## Graph statistics and mechanism signal

The clean graphs differ sharply enough to be useful pilot substrates:

| Dataset | Mean degree | Hubness max/mean | Positive-neighbor rate | Neighborhood noise | Feature-similarity lift |
|---|---:|---:|---:|---:|---:|
| WebQSP | 16.90 | 121.77 | 0.0324 | 0.9676 | 0.0564 |
| 2Wiki | 6.50 | 13.95 | 0.1327 | 0.8673 | 0.0389 |
| MuSiQue | 14.16 | 20.59 | 0.0338 | 0.9662 | 0.0267 |

At 25% rewiring, positive-neighbor rate falls to 0.0264, 0.0994, and 0.0253;
feature-similarity lift falls to 0.0377, 0.0285, and 0.0181. Degree and hubness
remain unchanged by construction. This is consistent with the intended causal
test: weaken neighborhood task/feature alignment while preserving the degree
distribution. It is a mechanism hypothesis, not yet a causal conclusion,
because the run has one seed and only one corruption level.

`edge_type_entropy` is deliberately unavailable, not zero: these pilot exports
have no trustworthy relation types. Typed-edge removal must hard-fail until
canonical artifacts include real edge semantics.

## What this pilot establishes

It establishes that the standalone repository can export neutral artifacts
from CRAG, run isolated parallel Modal jobs, enforce topology-only treatment,
match parameters and analytical compute, audit perturbation strength, collect
per-query structural statistics, and reproduce invariant MLP outputs across a
topology-only intervention.

It does **not** establish that GCNs win, that MLPs win, that rewiring causes the
observed differences in expectation, or that the resulting boundary transfers
across datasets. It cannot be cited in a submission as a main result.

## Next-run gate

Run only the following next:

1. degree-preserving rewiring at 0.10, 0.50, and 1.00 on the same three pilot
   datasets and exact protocol;
2. verify monotonic changes in positive-neighbor lift and inspect R@1/R@5/R@20
   separately;
3. if the response is coherent, run registered random-edge addition; otherwise
   debug the topology operator or training stability before expanding;
4. do not start hub injection, feature degradation, typed-edge removal, the
   full dataset suite, or the LODO predictor yet.

Before any paper-valid run, replace all three caches with complete canonical
train/validation/test data, freeze dataset manifests and checksums, tune on
validation only, use five seeds, include GCN/SAGE/GAT architecture robustness,
run parameter-, analytical-compute-, and equal-time comparisons, and record
VRAM/energy-equivalent budgets. Only then should the observed regimes be used
to build the phase diagram or train a held-out-dataset predictor.
