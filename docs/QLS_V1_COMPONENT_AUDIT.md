# What the historical baselines actually contain, and whether D2 duplicates one

Before spending anything on D2 -- *does QLS-v1's query-local structural block
carry ranking value?* -- the frozen ladder was audited to see whether that
experiment already exists. Names were not trusted. Every cell below was read
from the model constructor, the batch-preparation path, or the frozen result
file.

**Verdict: D2 is not a duplicate.** The historical `seed-only -> QLS-v1` gap of
**+2.58 R@5** on 2Wiki does not isolate the query-local block, because
`seed_only` never receives that block *or* the static block. It receives a
single binary column. The +2.58 bundles three changes at once.

**But one half of D2 is already answered, for free.** D0c's `RETRIEVAL_ONLY`
versus `CAND` *is* exactly "linear QLS minus the ten query-local columns", and
it is already measured. Only the nonlinear arm is missing.

---

## 1. Component matrix

`P` = projection dim (64). Rows are what the code builds, not what the name
suggests.

| | `plain_mlp` | `seed_only` | linear QLS (A3) | QLS-v1 (`sa_mlp`) | D0c `RETRIEVAL_ONLY` | **proposed D2 `NO_QUERY_LOCAL`** |
|---|---|---|---|---|---|---|
| Dense/SPLADE scalar retrieval features | no | no | **yes**, 2 reciprocal ranks | no | **yes**, 2 | no |
| seed indicator / seed features | no | **yes**, 1 binary column | via `seed_connections` + distance one-hot | via the local block only | via `seed_connections` + distance one-hot | **removed with the block** |
| candidate semantic embeddings | **yes** | **yes** | no | **yes** | no | **yes** |
| q/x learned projection | **yes**, 2x1536x64 | **yes**, 2x1536x64 | no | **yes**, 2x1536x64 | no | **yes**, identical |
| query x candidate interaction block | no | **yes**, 2P+2 | no | **yes**, 2P+2 | no | **yes**, identical |
| 7 static features | no | **no** (`static_dim=0`) | **yes** | **yes** | **yes** | **yes**, kept |
| 10 query-local structural features | no | **no** (`local_dim=1`, the seed bit) | **yes** | **yes** | **no** | **no** -- zeroed |
| PPR | no | no | **yes**: global `pagerank` (static) + `personalized_pagerank` (local) | **yes**, both | global `pagerank` only | global `pagerank` only |
| graph access | none | **none** | sealed feature cache | sealed feature cache | rebuilt from `graph.pt` | same as QLS-v1 |
| architecture | `PlainMLP`, query-only residual head | `ExplicitFeatureMLP` | linear, bias-free | `ExplicitFeatureMLP` | linear, bias-free | `ExplicitFeatureMLP`, **byte-identical to QLS-v1** |
| parameter count | 204,928 | 213,574 (head 65) | 19 | 213,506 (head 61) | 9 | **213,506** |
| loss | operator-screen listwise | multi-positive listwise CE | `segmented_listwise_loss` | multi-positive listwise CE | `segmented_listwise_loss` | multi-positive listwise CE |
| training procedure | operator screen runner | 5 seeds, test split | AdamW, 3 ep, bs 512q, lr 0.05, zeros init | 5 seeds, test split | AdamW, 10 ep, tail-selected | **identical to D1's CAND arm** |

VERIFIED FROM CODE: `src/mp_retrieval/operator_models.py:283-352` for what each
explicit-feature model admits, `scripts/run_sa_mlp_confirmation.py:198-208` for
what is actually handed to each at batch time, and `:466-470` for `seed_only`'s
`static_dim=0, local_dim=1`.

### The three things that make `seed_only` not the ablation

1. **It has no query-local block.** `_build_model` constructs it with
   `local_dim=1`, and `_prepare_batch` feeds it `_seed_indicator(...)` -- a
   single 0/1 column marking retrieval seed membership. The ten structural
   columns are never computed for it.
2. **It has no static block either.** `static_dim=0`. So `sa_mlp - seed_only`
   removes *both* structural blocks, not one.
3. **It gains a column QLS-v1 does not have.** The seed indicator is present in
   `seed_only` and absent from `sa_mlp`, whose seed information arrives only
   inside the local block.

So `sa_mlp - seed_only` = **+7 static, +10 query-local, -1 seed indicator**, at
two different head widths (61 vs 65, both parameter-matched to ~213.5k).

## 2. The historical +2.58 R@5, and why it cannot answer D2

2Wiki, five paired seeds, **test split** (1,500 queries). MEASURED, frozen.

| model | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---:|---:|---:|---:|---:|
| `plain_mlp` | 29.10 ± 0.39 | 50.25 ± 0.50 | 63.32 ± 0.42 | 75.33 ± 0.61 | 31.40 ± 0.73 |
| `seed_only` | 29.00 ± 0.78 | 65.83 ± 0.47 | 73.03 ± 0.21 | 79.52 ± 1.09 | 44.19 ± 0.41 |
| `sa_mlp` | 29.23 ± 0.97 | 68.40 ± 0.42 | 77.75 ± 0.10 | 79.66 ± 1.27 | 52.04 ± 0.22 |

`sa_mlp - seed_only`, per seed, points:

| seed | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---:|---:|---:|---:|---:|---:|
| 0 | +0.35 | +2.17 | +4.70 | +0.27 | +7.60 |
| 1 | -0.23 | +2.40 | +4.67 | -0.21 | +7.67 |
| 2 | -0.45 | +2.33 | +5.02 | -0.71 | +8.40 |
| 3 | +0.53 | +2.92 | +4.58 | +0.36 | +7.80 |
| 4 | +0.92 | +3.07 | +4.65 | +0.99 | +7.80 |
| **mean** | **+0.22** | **+2.58** | **+4.72** | **+0.14** | **+7.85** |

R@5 and R@20 are positive at every one of the five seeds and the Holm-corrected
p-values are 5.0e-4 and 1.6e-6; R@1 and MRR straddle zero (Holm p = 1).

**This establishes that the two structural blocks together matter.** It cannot
establish what the *query-local* block contributes on its own, because the
comparison also adds the seven static features and removes the seed indicator.

The one adjacent piece of evidence points the other way from a naive reading:
in the superseded seed-0 screen (`SA_MLP_SCREEN_RESULTS.md`, metaqa/webqsp/
hotpotqa -- **not 2Wiki**), the `static_structure` arm was *harmful* on all
three datasets while `query_local_structure` alone reached 30.02/28.29/73.22
against full SA's 30.04/32.66/77.22. That hints the local block carries most of
the combined effect, but it is a different dataset set, one seed, a superseded
stage, and its arms also drop the interaction block. It is a hint, not the
answer.

## 3. What is already answered: the linear half of D2

D0c ran `RETRIEVAL_ONLY` (2 rank + 7 static, 9 parameters) against `CAND`
(+ the 10 query-local columns, 19 parameters) on 2Wiki validation, converged,
tail-selected. Everything else identical. **That is exactly D2 at the linear
level**, and it was already paid for.

`CAND - RETRIEVAL_ONLY`, validation, points:

| R@1 | R@5 | R@20 | MRR |
|---:|---:|---:|---:|
| **-2.21** | **+0.96** | **+3.83** | **-2.85** |

MEASURED. At 19 parameters the query-local block **buys depth and costs
precision**: it is worth nearly four points of R@20 and one of R@5, and it
loses two of R@1 and nearly three of MRR.

That is a real and non-obvious result, and it is the reason D2's nonlinear arm
is worth running rather than assumed: a 213,506-parameter model with candidate
embeddings and an interaction block may or may not still need those ten columns,
and the linear answer does not transfer -- D0b already demonstrated what
happens when a 19-parameter result is used as a prior for the ranker.

## 4. Verdict

| question | answer |
|---|---|
| Is D2 already present in the frozen evidence? | **No.** |
| Does `seed_only -> sa_mlp` isolate the query-local block? | **No.** It bundles +7 static, +10 local, -1 seed indicator. |
| Is any other frozen condition `sa_mlp` minus only the local block? | **No.** The four `EXPLICIT_FEATURE_MODELS` are interactions-only, static-only, local-only, and all-of-it. "Interactions + static, no local" is not among them. |
| Is the linear version already answered? | **Yes**, by D0c, at no extra cost. |
| Does D2 therefore have decision value? | **Yes**, for the nonlinear arm only. |

D2 is filed as a two-arm ablation at 2Wiki x seed 0. It is **not** called
`RETRIEVAL_ONLY`: it keeps the seven static graph features and the full semantic
branch. It is named for what it removes.
