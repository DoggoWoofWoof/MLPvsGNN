P1c3p (docs/P1C3P_POINTWISE_HEAD.md): zrckp (the pointwise KT head) on KIND_T's pools against zrc on today's.

# scr-p1c3p-zrckp-pair re-called under the seed null: **PROMISING** (filed: MIXED)

R@5 of zrckp's p@swa minus zrc's p@swa of the same split (zrc's card fits on today's pools), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/P1C3P_POINTWISE_HEAD.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrc R@5 (today's) | zrckp R@5 (P1c3p) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7782 | 0.8019 | +0.0237 [+0.0207, +0.0265] | +0.0008, -0.0023 | 0.0075 | GAIN | GAIN |
| L-musique | squad | in-domain | 0.9105 | 0.9116 | +0.0012 [-0.0008, +0.0033] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5235 | 0.5121 | -0.0114 [-0.0198, -0.0033] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9041 | 0.9065 | +0.0025 [-0.0008, +0.0059] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.8733 | 0.9347 | +0.0614 [+0.0582, +0.0647] | -0.0002, -0.0007 | 0.0075 | GAIN | GAIN |
| L-musique | webqsp | zero-shot | 0.3384 | 0.3627 | +0.0243 [+0.0088, +0.0400] | -0.0070, -0.0018 | 0.0102 | GAIN | GAIN |
| L-hotpotqa | metaqa | in-domain | 0.7793 | 0.8044 | +0.0251 [+0.0222, +0.0279] | -0.0092, -0.0051 | 0.0149 | GAIN | GAIN |
| L-hotpotqa | squad | in-domain | 0.9112 | 0.9127 | +0.0015 [-0.0006, +0.0035] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5645 | 0.5745 | +0.0100 [+0.0021, +0.0176] | +0.0015, +0.0023 | 0.0075 | GAIN | GAIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8451 | 0.8513 | +0.0062 [+0.0020, +0.0106] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.8731 | 0.9350 | +0.0619 [+0.0587, +0.0652] | -0.0020, +0.0011 | 0.0075 | GAIN | GAIN |
| L-hotpotqa | webqsp | zero-shot | 0.2865 | 0.3475 | +0.0609 [+0.0430, +0.0776] | +0.0102, +0.0129 | 0.0233 | GAIN | GAIN |

Re-call: reads with a GAIN: 7 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> PROMISING; L-hotpotqa PROMISING -> PROMISING.
Changed calls: L-musique musique LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
