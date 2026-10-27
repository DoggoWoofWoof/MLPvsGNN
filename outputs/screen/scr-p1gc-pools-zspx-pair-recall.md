P1g-c (docs/P1GC_ADOPTED_ON_P1GB.md), the pools call: zspx (G5's GNN as filed) on P1g-b's pools against zspx on P1c2's pools.

# scr-p1gc-pools-zspx-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of zspx's p@swa minus zspx's p@swa of the same split (zspx's card fits on P1c2's pools), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/P1GC_ADOPTED_ON_P1GB.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zspx R@5 (P1c2's) | zspx R@5 (P1g-b's) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8030 | 0.7992 | -0.0038 [-0.0055, -0.0021] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9092 | 0.9131 | +0.0039 [+0.0019, +0.0060] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5073 | 0.5206 | +0.0133 [+0.0067, +0.0197] | +0.0503, -0.0079 | 0.0720 | GAIN | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9074 | 0.9105 | +0.0031 [+0.0006, +0.0057] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9351 | 0.9289 | -0.0062 [-0.0078, -0.0046] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3204 | 0.3513 | +0.0309 [+0.0170, +0.0444] | -0.0070, -0.0018 | 0.0102 | GAIN | GAIN |
| L-hotpotqa | metaqa | in-domain | 0.8039 | 0.7996 | -0.0043 [-0.0060, -0.0027] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9129 | 0.9120 | -0.0009 [-0.0028, +0.0008] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5901 | 0.5814 | -0.0087 [-0.0141, -0.0034] | +0.0015, +0.0023 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | hotpotqa | zero-shot | 0.8547 | 0.8575 | +0.0028 [-0.0003, +0.0059] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9312 | 0.9338 | +0.0026 [+0.0008, +0.0044] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2914 | 0.3072 | +0.0158 [+0.0025, +0.0293] | +0.0102, +0.0129 | 0.0233 | GAIN | **WITHIN** |

Re-call: reads with a GAIN: 1 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique PROMISING -> PROMISING; L-hotpotqa MIXED -> NO_GAIN.
Changed calls: L-musique musique GAIN -> WITHIN; L-hotpotqa webqsp GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
