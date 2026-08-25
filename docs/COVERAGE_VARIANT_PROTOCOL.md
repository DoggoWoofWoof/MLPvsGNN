# Preregistered K-direction set-coverage Offset variant

> **Status:** `PREREGISTERED_COVERAGE_VARIANT_GATE`  
> **Trigger:** confirmation result commit `c7813d0`  
> **Frozen before variant test:** 2026-08-25  
> **Controlling configuration:** `configs/coverage_variant.yaml`

## Why this single variant is permitted

The five-seed diagnosis found a monotonic Offset-minus-GNN R@20 loss as the
number of gold documents rises:

- 2Wiki K=4: -8.24 points for two golds and -10.65 for four;
- MuSiQue K=4: -3.49, -7.87, and -8.36 for two, three, and four golds.

The original K=4 model uses max-direction inference but trains through one
candidate-level listwise distribution. It does not require different directions
to cover different positives. The diagnostic therefore satisfies the frozen
condition for one coverage-aware objective. No architecture family, anchor,
candidate set, or inference rule changes.

## Frozen formulation

The model predicts four normalized relation targets:

\[
z_k = \operatorname{normalize}(a + g_k(q,a)), \quad k=1,\ldots,4,
\]

and directional candidate logits

\[
s_{ik}=\cos(x_i,z_k)/\tau.
\]

For a query with in-pool positive candidates \(P=\{p_1,\ldots,p_m\}\), where
\(m\leq4\), each direction defines a softmax over all common candidates. The
assignment loss is

\[
L_{set}=\min_{\pi\in\mathrm{Injective}(m,4)}
\frac{1}{m}\sum_{j=1}^{m}
-\log\operatorname{softmax}_{i}(s_{i,\pi(j)})_{p_j}.
\]

Thus every available positive is used and two positives cannot be assigned to
the same direction. Enumerating at most 24 assignments makes the hard minimum
exact and permutation-invariant.

Target collapse is penalized with

\[
L_{div}=\operatorname{mean}_{k\ne l}
\left[\max(0,\cos(z_k,z_l)-0.2)\right]^2,
\]

and the frozen objective is

\[
L=L_{set}+0.1L_{div}.
\]

There is no coefficient search. Queries with more than four in-pool positives
would fail the run rather than silently drop labels; audited 2Wiki/MuSiQue have
at most four.

Inference remains exactly

\[
s_i=\max_k s_{ik}.
\]

The model never reads adjacency during training or inference and has the same
221,504 parameters as the original K=4 Offset.

## Run and comparison

Run only this model on complete 2Wiki and MuSiQue for seeds 0-4, with the same
three epochs, candidates, loss negatives, optimizer, dropout, temperature,
splits, checkpoint selection by validation R@5, and repeated A10 inference
measurement as the confirmation. Compare within seed to the already frozen
original K=4 and GAT/GCN results from commit `c7813d0`.

## Success criteria

All criteria are frozen before seeing variant test output.

Primary mechanism success requires, on **both** datasets:

- at least +2.0 R@20 percentage points over original K=4; and
- a paired 95% confidence-interval lower bound above zero.

Also report:

- fraction of the original K=4-to-GNN R@20 gap closed, with 25% as the
  preregistered practical target;
- fraction of original K=4's R@1 advantage over the GNN retained, with 75% as
  the target;
- R@5, MRR, FullCov@20, latency, memory, and training cost;
- answer-count-stratified gains.

If the primary criterion fails, the mechanism is not supported by this
formulation. No alternative diversity weight, margin, assignment relaxation,
or second coverage model may be tried under this gate.
