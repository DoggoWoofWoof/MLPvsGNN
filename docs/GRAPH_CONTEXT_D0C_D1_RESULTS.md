# Stages D0c and D1: the stratification does not survive convergence

D0b reported a striking split. `TARGET_H1` and `CAND` were a wash overall, and
underneath, restoring context gained **+3.06** R@1 on queries whose hardest gold
was isolated inside `G[Cq]` and lost 1.7 to 2.5 points on every other stratum.
That result is what D1 was launched on, and it is the result these two stages
retract.

**Two independent stages, run to convergence, both say the same thing: the
restored one-hop context carries no ranking value this pipeline can use.**

- **D0c**, the same linear control at ten epochs instead of three, does not
  reproduce D0b's split. It **reverses** it. `TARGET_H1 - CAND` R@1 is now
  **-1.80** on the isolated stratum and **+3.28** on `ordinary` -- the sign of
  the effect flips on both ends.
- **D1**, the 213,506-parameter QLS-v1 ranker on the real embeddings, is null
  overall (R@5 67.88 -> 67.97) **and null in every pre-registered stratum**. The
  largest per-stratum move is +0.83 R@1 on `degree_1`; the largest anywhere is
  -1.01 R@20 on `ordinary`, which is 198 queries.

The pre-registered hypothesis -- *context value decreases as usable
candidate-induced structural evidence increases* -- **is not supported.** It was
recorded in `configs/graph_context_pilot.yaml` before either stage reported, on
strata frozen from D0b, and the result files carry `boundaries_fitted_here:
false`.

| | |
|---|---|
| **Decides** | `TARGET_H1` is not a QLS-v2 context. Null at the ranker that would use it, on the dataset with the most to gain. |
| **Decides** | D0b's stratified split was an artefact of comparing arms mid-ascent. It is retracted as evidence. |
| **Does not decide** | that `N1_in(Cq)` contains no structure. D0's mechanistic finding stands; what is dead is its incremental value to this ranker. |
| **Cost** | D0c 451.2 s / $0.08 CPU. D1 two runs, 427.3 s / $0.21 / 0.119 GPU-h, against 1.0 GPU-h and $2.50 authorised. |

---

## 1. D1: the experiment that was supposed to answer this

2wiki_clean, seed 0, `sa_mlp` at the frozen confirmation hyperparameters, fit on
9,450 train queries with a 1,050-query deterministic tail for epoch selection,
validation read once per arm. Two arms, one context each, identical in
everything else. MEASURED.

| arm | R@1 | R@5 | R@20 | MRR |
|---|---:|---:|---:|---:|
| `CAND` | **29.03** | 67.88 | **77.53** | 79.28 |
| `TARGET_H1` | 28.98 | **67.97** | 77.49 | **79.37** |

`TARGET_H1 - CAND`, validation, points, against the strata fixed in advance:

| stratum | n | R@1 | R@5 | R@20 | MRR |
|---|---:|---:|---:|---:|---:|
| isolated | 834 | -0.48 | -0.06 | +0.09 | -0.28 |
| degree_1 | 750 | +0.83 | +0.23 | +0.07 | +0.78 |
| low_degree | 1218 | -0.27 | +0.14 | -0.02 | -0.12 |
| ordinary | 198 | -0.25 | -0.25 | -1.01 | +0.28 |
| overall | 3000 | -0.05 | +0.08 | -0.03 | +0.09 |

There is no trend. `non_increasing` holds for R@20 alone, and it holds there by
sliding from +0.09 to -1.01 across strata whose deltas are all inside a point.
On the isolated stratum -- the 27.8% of queries the whole line was aimed at, and
where D0b measured +3.06 R@1 -- the ranker gains **nothing**: -0.48 R@1,
-0.06 R@5.

### The run reproduced exactly

D1 ran twice: once as declared, and once more after the stratified reporting was
added, because the first run emitted only the overall numbers. Both runs, in
separate containers, returned **bit-identical** validation metrics for both arms
on all four measures. The second run added reporting and changed no protocol
value. MEASURED, and it is also the cheapest determinism check this pipeline has.

The strata reconstitute the overall metric exactly under their own query counts;
that identity is enforced by a test rather than asserted here.

## 2. D0c: what three epochs was hiding

Six arms, ten epochs, epoch chosen on the train tail, validation read once each.
EXPLORATORY -- the threshold in the selective arms was chosen after seeing D0b's
strata, and none of this is confirmatory.

**Every arm's selected epoch is interior** -- 4 to 9 out of 10, none at the
boundary -- and every arm's train-tail R@5 peaked and turned down inside the
budget. That is the property D0b lacked, where all four arms were still climbing
when the three-epoch budget ended. It is not the same as full convergence: the
mean train loss over the last four epochs is still creeping by 4.3% for
`RETRIEVAL_ONLY` and 3.2% for `SELECTIVE_MASK`, the two arms without the full
local block, against under 1.4% for the four that have it. `TARGET_H1` selects
epoch 9 of 10, which is interior by one epoch and is the weakest such claim
here.

| arm | epoch | R@1 | R@5 | R@20 | MRR |
|---|---:|---:|---:|---:|---:|
| `RETRIEVAL_ONLY` | 5 | **39.86** | 67.42 | 71.58 | **94.45** |
| `CAND` | 7 | 37.65 | **68.38** | 75.40 | 91.61 |
| `TARGET_H1` | 9 | 37.78 | 68.23 | 75.38 | 91.97 |
| `BOTH` | 6 | 36.69 | **68.38** | **75.63** | 90.48 |
| `SELECTIVE_SUBSTITUTE` | 8 | 36.54 | 68.29 | 75.50 | 90.41 |
| `SELECTIVE_MASK` | 4 | 31.35 | 67.66 | 71.83 | 83.89 |

D0c fits on train-minus-tail where D0b fit on all of train, so these absolutes
are not a continuation of D0b's and are not compared to them. The arms are
compared to each other.

### The sign flip

`TARGET_H1 - CAND` at ten epochs, beside the same quantity at D0b's three:

| stratum | n | R@1 3ep | R@1 10ep | MRR 3ep | MRR 10ep |
|---|---:|---:|---:|---:|---:|
| isolated | 834 | **+3.06** | **-1.80** | +4.44 | -2.02 |
| degree_1 | 750 | -1.70 | -0.93 | -2.26 | -1.16 |
| low_degree | 1218 | -2.50 | +1.60 | -3.18 | +2.32 |
| ordinary | 198 | -0.51 | +3.28 | -1.17 | +4.15 |

MEASURED. Both ends invert. At convergence the restored context helps the
*well-connected* candidates and hurts the starved ones, which is the reverse of
the reading D1 was launched to test. Neither shape survives into D1, so the
right conclusion is not "the opposite is true" -- it is that a 19-parameter model
compared mid-ascent was measuring its own rate of ascent.

### Conditional use is not promising

This is what D0c was approved to find out, and the answer is no.

- `BOTH` -- both blocks, coefficients chosen by the learner -- lands on R@5
  **68.38**, equal to `CAND` to the digit, and loses 0.96 R@1. Handed the
  restored context alongside the historical one, the learner extracts nothing
  from it.
- `SELECTIVE_SUBSTITUTE` -- restored context exactly where induced degree < 2 --
  is **worse** than `CAND` on three of four measures (-1.11 R@1, -1.20 MRR).
- `SELECTIVE_MASK` -- restored context on starved rows, zeros elsewhere -- is the
  worst arm by a wide margin (-6.30 R@1, -7.72 MRR), which confirms the block is
  load-bearing: deleting historical context from connected rows destroys real
  information.

**A principled query-conditioned reliability feature is not justified by this
evidence.** The exploratory question was whether conditional use looked promising
enough to fund that design. It does not. No learned gate was built, and none
should be.

## 3. `RETRIEVAL_ONLY` at convergence

D0b's +6.12-point R@1 lead for the nine-feature arm was flagged as not a valid
feature conclusion, because every arm was still climbing. Converged, the lead
**shrinks but does not vanish**: +2.21 R@1 and +2.85 MRR over `CAND`, against
-0.96 R@5 and -3.83 R@20.

So roughly two thirds of that gap was the unconverged comparison and one third
is real at this model size. The structural block still buys depth and still costs
top-of-list precision. What that means for QLS-v1 is untested -- D1 ran no
`RETRIEVAL_ONLY` arm -- and it is the open question these stages leave behind.

## 4. What this does and does not decide

- **Decides:** `TARGET_H1` does not enter QLS-v2 as a context. Null at the ranker,
  on the dataset with the highest isolated-gold prevalence of the six.
- **Decides:** D0b's stratified split is retracted. It does not reproduce at
  convergence and does not appear at the ranker.
- **Decides:** no learned gate, no attention, no mixture of experts on this
  evidence. Conditional use was tested directly and was not promising.
- **Does not decide:** that the ten local columns are worth their place. D0c says
  they trade R@1 for R@20 at 19 parameters; nobody has run that comparison at
  213,506.
- **Does not decide:** anything at a second dataset or a second seed. One seed,
  one dataset, one context definition.
