# UMLP-v2.1 — coverage-aware objective (stage A)

Declaration: `configs/umlp_v21_coverage_objective.yaml` (filed bfea912, before any fit). Runner: `scripts/umlp_v21_coverage.py` (0a2f0c8).

**Reading: V21_STAGE_A_FAIL. v2.1 stops here.** λ, loss, features, architecture and thresholds are unchanged after the result; seeds 1–2 and stage B are not run.

## What changed

Only the training objective: L = listwise + 1.0 · coverage. The coverage term, for queries with at least two in-pool golds, is the mean over golds g and the five highest-scoring in-pool non-golds n (chosen on detached scores) of softplus(s_n − s_g). It is summed and divided by the number of queries that have an in-pool gold.

Everything else was frozen and checked against the control's seed-0 fit record before the fit, field by field:
- architecture `u_mlp_v2_mix`, 330,955 parameters;
- the 129-column contract (core `8d1da88b…`);
- the relation bank;
- the training rule (6 epochs max, 2,000 batches of 16, AdamW 1e-3 / 1e-4, clip 1.0, patience 2, per-query draw);
- seed 0, 8 threads, 2 pack workers at depth 2.

## Fit

`u_mlp_v21_cov__H128__s0`: best epoch 1, select macro R@5 0.8470. Early-stopped after epoch 3 (patience 2), 6,207 s of fit time.

| epoch | train loss | select macro R@5 | 2wiki | metaqa | squad |
|---|---|---|---|---|---|
| 0 | 2.0259 | 0.8457 | 0.886 | 0.733 | 0.919 |
| 1 | 1.4984 | **0.8470** | 0.884 | 0.747 | 0.910 |
| 2 | 1.2518 | 0.8414 | 0.885 | 0.749 | 0.891 |
| 3 | 1.1042 | 0.8430 | 0.879 | 0.750 | 0.900 |

The control's best was epoch 0, at 0.8457.

**Incident (systems only):** Claude Code's memory-pressure safeguard stopped the background process three times: during epoch 1, epoch 2 and epoch 3. Each relaunch resumed through the frozen `fit_model` checkpoint. The checkpoint holds weights, AdamW state, both RNG states, the per-dataset draw cursors, the best weights, the patience counter and the record. There is no LR scheduler in the rule.

Before the first resume, the checkpoint was checked:
- arm, seed and config match;
- epochs_done 1, 2,000 steps, AdamW step 2000 on all 14 tensors;
- 0 skipped batches.

Each resume logged "resumed from … after epoch k" and continued at epoch k+1. Nothing scientific was changed, and the safeguard was not disabled.

## Stage-A gate — 2wiki V2_GATE (6,290 queries)

The frozen control seed 0 was re-scored in the same pass and reproduced every stored per-query metric (0 mismatches). Paired bootstrap: 1,000 resamples, default_rng(0), 95%, v2.1 minus control.

| metric | control s0 | v2.1 s0 | Δ mean | 95% CI | condition | holds |
|---|---|---|---|---|---|---|
| full_coverage@5 | 0.6752 | 0.6763 | +0.0011 | [−0.0084, +0.0094] | lower > 0 | no |
| recall@5 | 0.8545 | 0.8536 | −0.0010 | [−0.0052, +0.0029] | mean ≥ 0 | no |
| hit@1 | 0.9107 | 0.8830 | −0.0277 | [−0.0343, −0.0207] | mean ≥ −0.005 | no |

Other metrics (descriptive only):

| metric | Δ mean | 95% CI |
|---|---|---|
| recall@1 | −0.0126 | [−0.0157, −0.0095] |
| recall@10 | −0.0045 | [−0.0077, −0.0013] |
| recall@20 | −0.0025 | [−0.0045, −0.0005] |
| mrr | −0.0156 | [−0.0193, −0.0118] |
| ndcg@5 | −0.0066 | [−0.0101, −0.0036] |
| ndcg@20 | −0.0075 | [−0.0099, −0.0052] |
| full_coverage@20 | −0.0040 | [−0.0084, +0.0003] |

## Interpretation

The coverage objective did not move set coverage. The FullCov@5 change was +0.001 with an interval centred on zero, against a D0 gap to message passing of −0.078. It cost first-gold ranking: hit@1 fell by 2.8 points, and the whole interval lies below the filed tolerance.

D0.2 showed that a probe *fitted to the D0.1 pairs* separates the GNN-recovered golds from the displacers in the frozen columns. Pushing golds above the current top-5 non-golds during training did not turn that separability into coverage. Instead the scorer traded top-1 precision for nothing measurable at k=5.

Scope: one seed, one λ, one objective form, 2wiki V2_GATE only. This is a failed pre-registered stage-A gate. It does not show that no objective could close the gap. MetaQA and SQuAD were not evaluated. The held half was not read.

For the registered question, the 2wiki coverage share of the message-passing gap stays unexplained by this objective change: the matched non-message-passing scorer, retrained with an explicit coverage objective, did not recover it.
