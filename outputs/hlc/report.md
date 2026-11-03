# HLC: the hyperlinks' lift to our models against their lift to HippoRAG 2's walk

Declared in docs/HLC_HYPERLINK_LIFT_MODELS_VS_WALK.md. Calls: family H **GREATER_EVERYWHERE**, family W **MIXED**. R@5 points, paired by question, 95% intervals (2000 resamples).

## Family H: the corpus's own hyperlinks (U1a's b1 against b1t; the walk's SNK against NK)

| setting | model | read | model lift | walk lift | difference | call | model with − walk with |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2wiki | zsp/J5 (GNN) | in-domain | +19.8 [+18.3, +21.2] | -0.1 [-0.9, +0.6] | +20.0 [+18.4, +21.5] | GREATER | +35.3 [+33.6, +37.0] |
| 2wiki | zsp/L-2wiki (GNN) | held-out | +16.3 [+15.0, +17.6] | -0.1 [-0.9, +0.6] | +16.5 [+14.9, +17.9] | GREATER | +31.6 [+29.9, +33.2] |
| 2wiki | zrc/J5 (MLP) | in-domain | +20.8 [+19.3, +22.2] | -0.1 [-0.9, +0.6] | +20.9 [+19.3, +22.5] | GREATER | +34.5 [+32.8, +36.2] |
| 2wiki | zrc/L-2wiki (MLP) | held-out | +17.7 [+16.4, +19.1] | -0.1 [-0.9, +0.6] | +17.9 [+16.4, +19.3] | GREATER | +31.5 [+29.8, +33.1] |
| hotpotqa | zsp/J5 (GNN) | in-domain | +8.2 [+7.0, +9.4] | +5.1 [+4.1, +6.3] | +3.1 [+1.4, +4.7] | GREATER | +17.5 [+15.8, +19.1] |
| hotpotqa | zsp/L-hotpotqa (GNN) | held-out | +6.8 [+5.7, +7.9] | +5.1 [+4.1, +6.3] | +1.6 [+0.0, +3.2] | SAME | +15.2 [+13.5, +16.8] |
| hotpotqa | zrc/J5 (MLP) | in-domain | +9.4 [+8.2, +10.7] | +5.1 [+4.1, +6.3] | +4.3 [+2.6, +5.9] | GREATER | +17.4 [+15.7, +19.1] |
| hotpotqa | zrc/L-hotpotqa (MLP) | held-out | +8.0 [+6.8, +9.2] | +5.1 [+4.1, +6.3] | +2.9 [+1.1, +4.5] | GREATER | +15.8 [+14.1, +17.5] |
| musique | zsp/J5 (GNN) | in-domain | +3.3 [+2.1, +4.5] | +0.8 [-0.1, +1.8] | +2.5 [+1.1, +3.9] | GREATER | +16.3 [+14.6, +17.9] |
| musique | zsp/L-musique (GNN) | held-out | +2.5 [+1.0, +3.9] | +0.8 [-0.1, +1.8] | +1.7 [-0.1, +3.2] | SAME | +10.3 [+8.4, +12.1] |
| musique | zrc/J5 (MLP) | in-domain | +2.7 [+1.6, +3.8] | +0.8 [-0.1, +1.8] | +1.9 [+0.4, +3.3] | GREATER | +16.2 [+14.5, +17.8] |
| musique | zrc/L-musique (MLP) | held-out | +2.9 [+1.6, +4.3] | +0.8 [-0.1, +1.8] | +2.1 [+0.6, +3.6] | GREATER | +11.4 [+9.4, +13.1] |

## Family W: the recreated links W2_S (B1P's b1w against b1t; the walk's W2_S+NK against NK)

| setting | model | read | model lift | walk lift | difference | call | model with − walk with |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2wiki | zg6r/in_domain (GNN) | in-domain | +16.8 [+15.5, +18.2] | +0.1 [-0.8, +0.9] | +16.8 [+15.2, +18.3] | GREATER | +32.9 [+31.1, +34.6] |
| 2wiki | zrckp/in_domain (MLP) | in-domain | +18.1 [+16.8, +19.4] | +0.1 [-0.8, +0.9] | +18.0 [+16.5, +19.4] | GREATER | +31.9 [+30.2, +33.8] |
| hotpotqa | zg6r/in_domain (GNN) | in-domain | +3.1 [+2.4, +4.0] | +3.4 [+2.4, +4.3] | -0.2 [-1.5, +1.0] | SAME | +15.4 [+13.8, +17.0] |
| hotpotqa | zrckp/in_domain (MLP) | in-domain | +3.4 [+2.5, +4.3] | +3.4 [+2.4, +4.3] | +0.0 [-1.2, +1.2] | SAME | +14.5 [+12.9, +16.1] |
| hotpotqa | zg6r/zero_shot (GNN) | zero-shot | +3.5 [+2.6, +4.4] | +3.4 [+2.4, +4.3] | +0.1 [-1.1, +1.3] | SAME | +14.0 [+12.4, +15.5] |
| hotpotqa | zrckp/zero_shot (MLP) | zero-shot | +4.3 [+3.4, +5.2] | +3.4 [+2.4, +4.3] | +1.0 [-0.3, +2.2] | SAME | +13.3 [+11.7, +14.9] |
| musique | zg6r/in_domain (GNN) | in-domain | +3.0 [+1.8, +4.0] | +1.0 [+0.1, +1.9] | +2.0 [+0.6, +3.3] | GREATER | +18.8 [+17.1, +20.5] |
| musique | zrckp/in_domain (MLP) | in-domain | +4.4 [+3.3, +5.6] | +1.0 [+0.1, +1.9] | +3.4 [+2.1, +4.9] | GREATER | +16.8 [+15.1, +18.5] |
| musique | zg6r/zero_shot (GNN) | zero-shot | -1.3 [-2.6, +0.1] | +1.0 [+0.1, +1.9] | -2.3 [-3.8, -0.8] | LESS | +10.7 [+8.8, +12.5] |
| musique | zrckp/zero_shot (MLP) | zero-shot | +2.3 [+0.9, +3.5] | +1.0 [+0.1, +1.9] | +1.2 [-0.2, +2.7] | SAME | +10.8 [+8.9, +12.5] |

## Levels (R@5)

| setting | R0 | walk NK | walk SNK | walk W2_S+NK | walk W2_S+NK (stored) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2wiki | 0.7117 | 0.6198 | 0.6182 | 0.6205 | 0.613 |
| hotpotqa | 0.85 | 0.7555 | 0.807 | 0.789 | 0.8005 |
| musique | 0.5643 | 0.491 | 0.4994 | 0.5012 | 0.4996 |

Checks: {'hlw_arms': True, 'hlw_add': True, 'u1a_paired': True, 'b1p_paired': True, 'read_sha256': True, 'read_ids': True, 'w2s_bench_shares_b1': True}.
