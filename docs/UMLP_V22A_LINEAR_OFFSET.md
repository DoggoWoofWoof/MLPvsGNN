# UMLP-v2.2A — frozen-base linear score offset (stage A)

Declaration: `configs/umlp_v22a_linear_offset.yaml` (2a5e592, filed before any fit). Runner: `scripts/umlp_v22a_offset.py` (8e50875).

**Reading: V22A_STAGE_A_FAIL. v2.2A stops here.** τ, λ_p, λ_Δ, the pair definition, features and thresholds are unchanged after the result. Seeds 1–2 and stage B are not run.

## Model

The frozen `u_mlp_v2_mix__H128__s0` is always in eval mode, and its weights are byte-identical after the fit (digest `b8d40090…`).

The score is ŝ = z_q(s0) + τ·tanh(w·z_q(x)):
- z_q is the within-query z-score;
- x is the twin's 129 read columns;
- w has 129 weights and starts at 0, so the model starts identical to the twin;
- τ = 1.

The objective combines three terms:
- **L_secondary:** softplus of secondary golds against the twin's top-5 non-golds, on multi-gold queries.
- **L_protect** (λ_p = 1): relu on any shrinkage of the first gold's margin.
- **L_size** (λ_Δ = 0.01): mean Δ².

Training used the frozen rule and selection.

## Fit

Best epoch 4, select macro R@5 0.8463 (the twin alone scores 0.8457). The run went 6 epochs, 3,685 s. The host memory safeguard stopped it once during epoch 3; it resumed from the full-state checkpoint.

| epoch | train loss | select macro R@5 |
|---|---|---|
| 0 | 0.4996 | 0.8441 |
| 1 | 0.4981 | 0.8440 |
| 2 | 0.4924 | 0.8460 |
| 3 | 0.4878 | 0.8459 |
| 4 | 0.4880 | **0.8463** |
| 5 | 0.4857 | 0.8454 |

The objective barely moved: it fell by 0.014 over six epochs. The largest learned weights were:

| column | weight |
|---|---|
| has_typed_edge | +1.86 |
| relmean_in | +0.77 |
| cos_v_seedproto | −0.57 |
| relmax_in | +0.53 |
| dense_cos | +0.47 |
| relchain2_max | +0.42 |

‖w‖ = 2.54. The weights are dominated by relation and typed-edge columns, which carry signal on the KB set (MetaQA) and are mostly constant, hence zero after z-scoring, on 2wiki passages. One w is shared across the trio.

## Stage-A gate — 2wiki V2_GATE (6,290 queries)

The frozen control was re-scored in the same pass and reproduced every stored per-query metric (0 mismatches). Paired bootstrap: 1,000 resamples, default_rng(0), 95%, offset minus control.

| metric | control | v2.2A | Δ mean | 95% CI | condition | holds |
|---|---|---|---|---|---|---|
| full_coverage@5 | 0.6752 | 0.6603 | −0.0149 | [−0.0224, −0.0078] | lower > 0 | no |
| recall@5 | 0.8545 | 0.8459 | −0.0086 | [−0.0124, −0.0054] | mean ≥ 0 | no |
| hit@1 | 0.9107 | 0.9188 | +0.0081 | [+0.0038, +0.0127] | mean ≥ −0.005 | yes |

Other metrics (descriptive only):

| metric | Δ mean | 95% CI |
|---|---|---|
| recall@1 | +0.0039 | [+0.0019, +0.0060] |
| mrr | +0.0040 | [+0.0015, +0.0064] |
| ndcg@5 | −0.0038 | [−0.0062, −0.0015] |
| recall@10 | −0.0041 | [−0.0070, −0.0013] |
| full_coverage@20 | −0.0046 | [−0.0086, −0.0008] |

Size of the offset:
- mean |Δ̂| is 0.40 within-query standard deviations;
- the top-1 changed on 13.1% of queries;
- the top-5 set changed on 70.6% of queries.

## Interpretation

The offset moved the ranking substantially: 71% of top-5 sets changed. But it moved it toward the first gold, not the secondary golds. Hit@1 rose by 0.8 points, while coverage fell by 1.5 points. The protection term and the first-gold-heavy trio selection won; the secondary-pair term hardly decreased in training.

Together with v2.1 (coverage objective on the whole scorer: FC@5 +0.001, Hit@1 −0.028), two differently parameterised attempts to turn D0.2's linear separability into 2wiki coverage have both failed. One moved nothing on coverage and hurt the top; the other helped the top and hurt coverage.

The most economical reading is D0.2's own caveat. The pairs it separated were conditioned on "the GNN recovered this gold". Separability on that selected population does not carry over to a trained, unconditioned correction.

Scope: one seed, one τ, one linear form, 2wiki V2_GATE only. MetaQA and SQuAD were not evaluated. The held half was not read.

For the registered question, the coverage share of the 2wiki message-passing gap remains unrecovered by a matched non-message-passing scorer under two pre-registered corrections over the same compiled basis.
