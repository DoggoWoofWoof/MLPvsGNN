# Frozen five-seed confirmation and capacity protocol

> **Status:** `CONFIRMATION_GATE_NOT_PAPER_FINAL`  
> **Frozen before launch:** 2026-08-25  
> **Ancestor:** `paper-protocol-v0`; screen implementation `5a6b8e3`  
> **Controlling configuration:** `configs/confirmation.yaml`

## Purpose

The one-seed screen did not support a general Offset-MLP-over-GNN claim. This
gate tests two narrower observations without expanding architectures or
datasets:

1. Does plain MLP's R@5 result on 2Wiki and MuSiQue replicate across seeds?
2. Can a substantially smaller topology-free model retain that effectiveness?

It also tests whether Offset's loss at R@20 grows with the number of relevant
documents, as predicted by a sharp-first-answer versus set-coverage mechanism.

## Frozen datasets and models

Only complete 2Wiki and MuSiQue data are used. For seeds 0-4, confirm the
current-width models:

- plain MLP, hidden width 64;
- Offset-MLP K=1, hidden width 64;
- Offset-MLP K=4, hidden width 64;
- the GNN selected before this experiment using the original screen's
  validation R@5: GAT for 2Wiki and GCN for MuSiQue, hidden width 64.

No GNN may be replaced after seeing confirmation test metrics.

In the same run, sweep widths 16, 32, and 64 for plain MLP and K=1 Offset. The
widths were chosen before launch because they produce approximately 25%, 50%,
and 100% of the current parameter count. Plain MLP has 49,696/100,416/204,928
parameters and K=1 Offset has 49,952/101,440/209,024. K=4 and the frozen GNN
remain at width 64.

## Capacity selection

Capacity is selected separately for plain and Offset in each dataset using mean
validation R@5 across seeds 0-4. Test performance is not an input. If multiple
widths fall within 0.1 percentage point of the best validation mean, choose the
smallest parameter count. All capacity points remain visible, but only this
validation-selected capacity may support the parameter-efficiency claim.

## Controls and measurements

The complete-data comparison contract is unchanged: identical frozen query and
node embeddings, dense+SPLADE candidate union and order, labels, eligible
training queries, listwise multi-positive loss, optimizer, epochs, batches,
splits, and seeds. Candidate-induced adjacency remains the GNN's only
privileged input. Offset models never access it.

For each model/seed retain R@1/R@5/R@20, MRR, FullCov, conditional metrics,
parameters, training time, and five repeated synchronized inference
measurements. Report mean, sample standard deviation, paired seed differences,
and 95% confidence intervals. Screening seed 0 is rerun so this result is
self-contained rather than spliced from an older execution.

## Coverage diagnosis

Report the test metrics and paired Offset-minus-GNN gaps for each observed
number of gold documents. Test whether the R@20 gap becomes monotonically more
negative as the gold set grows; this is a diagnosis, not a selection criterion.

The frozen manifests contain empty per-query hop values. MuSiQue hop count is
recoverable for every query from the read-only raw question-decomposition
length. 2Wiki's processed records contain no per-query hop label, so its hop
analysis must be reported as unavailable rather than inferred from gold count.
If MuSiQue hop count and gold count are identical, their effects are explicitly
non-identifiable in this substrate.

## Conditional coverage-aware variant

No new Offset variant is part of the confirmation run. It may be implemented
only if the five-seed stratified diagnosis supports multi-target failure. Its
loss, assignment, diversity penalty, and coefficients must then be frozen in a
new commit before any variant test metrics are observed.

No topology perturbation, new dataset, extra GNN, or architecture search may
launch before the confirmation report is complete.
