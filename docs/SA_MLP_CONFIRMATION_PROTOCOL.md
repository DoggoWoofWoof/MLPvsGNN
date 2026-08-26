# SA-MLP fairness-controlled confirmation protocol

Status: **frozen after the one-seed gate and before any new confirmation test
metric is computed**. The annotated protocol tag records the immutable code,
configuration, and analysis contract used for all new runs.

## Why this control is required

The registered screen passed on MetaQA, WebQSP, and HotpotQA, but its
query-local feature vector contains a distance-0 bucket. Distance 0 is exactly
membership in the stable union of frozen dense top-5 and SPLADE top-5 seeds.
The screen's old selected GNN did not receive that binary indicator. The screen
therefore demonstrates that the whole fixed-feature package is effective; it
does not yet establish that paths or PPR, rather than the frozen retriever
prior, caused the gain.

No screen feature, normalization, projection, head-width rule, training loss,
candidate, label, split, or selected GNN family may change during confirmation.

## Models and causal contrasts

The primary matched-parameter regime contains three newly trained models:

1. **SA-MLP:** the exact frozen screen model with interaction, seven static,
   and ten query-local inputs. Its learned forward receives no adjacency.
2. **Seed-only:** the same interaction scorer plus one binary indicator for
   frozen dense/SPLADE seed membership. It receives neither graph descriptors
   nor adjacency. This measures the retriever-prior explanation.
3. **Seed-aware GNN:** the already validation-selected family for each dataset
   with the same binary seed indicator concatenated before its original node
   projection. This changes only the projection from 1,536 to 1,537 inputs and
   adds exactly 64 parameters. It receives candidate-induced adjacency.

The frozen plain MLP and frozen selected GNN remain reference baselines and are
not retrained. The decisive contrasts are SA-MLP minus seed-only, SA-MLP minus
seed-aware GNN, and seed-aware GNN minus frozen GNN.

## Datasets, seeds, and reuse

Run all five canonical seeds on 2Wiki, MuSiQue, WebQSP, HotpotQA, SQuAD, and
all 407,513 MetaQA queries. For MetaQA, WebQSP, and HotpotQA, reuse the exact
SA-MLP seed-0 checkpoint result already produced by the frozen screen after
verifying the candidate contract, data fingerprint, result hash, parameter
count, and feature contract. All other model/seed combinations are new.

Each paired run uses the same node/query arrays, candidate IDs, multi-positive
labels, listwise loss, negative pool, split, seed, optimizer, three-epoch
budget, and validation-only checkpoint rule. Test is evaluated once.

## Statistical contract

R@5 is primary. R@1, R@20, MRR, and FullCov@20 are mandatory secondary
outcomes. Report every dataset regardless of sign.

- paired five-seed Student-t intervals summarize optimizer variation;
- query-level paired intervals use a stratified bootstrap with seed as the
  outer block;
- Holm correction is applied across six datasets separately for each primary
  contrast;
- one absolute R@5 point is the preregistered non-inferiority margin.

The graph-summary mechanism is supported only if SA-MLP exceeds seed-only with
a Holm-adjusted positive R@5 interval on at least two of the three original
GNN-win datasets. If seed-only recovers at least 80% of SA-MLP's gain over the
plain MLP on two of those datasets, the seed-prior explanation wins and the
graph-path claim is rejected. Fixed summaries substitute for message passing
only where SA-MLP is non-inferior to the seed-aware GNN within one point.

## Systems contract

Online latency uses a warmup batch and median of five complete test passes at
batch size 16. Report throughput, training seconds, total and incremental GPU
allocation, total and incremental process RSS, topology build/cache cost, and
fixed-feature build/cache cost. Preprocessing is never hidden inside online
latency. Packed topology and packed feature caches are reused across seeds.

The practical-width regime is a separate validation-only extension using
projection widths 16/32/64. It is deliberately not mixed into this primary
confirmation; no test-selected efficiency claim is permitted.

## Interpretation boundary

A positive result would support: **retrieval benefits from graph information,
while fixed query-conditioned summaries can sometimes replace learned neighbor
aggregation.** It would not establish that graphs are unnecessary, that MLPs
always beat GNNs, or that preprocessing is free. A negative result remains
informative by locating the screen's gain in the frozen retriever prior.
