P1c3 (docs/P1C3_KT_INPUT.md): zspk (KT input) on KIND_T's pools against zsp on today's.

# scr-p1c3-zspk-pair re-called under the seed null: **PROMISING** (filed: MIXED)

R@5 of zspk's p@swa minus zsp's p@swa of the same split (zsp's card fits on today's pools), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/P1C3_KT_INPUT.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zsp R@5 (today's) | zspk R@5 (P1c3) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7794 | 0.8009 | +0.0215 [+0.0185, +0.0243] | +0.0008, -0.0023 | 0.0075 | GAIN | GAIN |
| L-musique | squad | in-domain | 0.9125 | 0.9114 | -0.0011 [-0.0032, +0.0012] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5111 | 0.5012 | -0.0100 [-0.0175, -0.0024] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9065 | 0.9080 | +0.0015 [-0.0016, +0.0047] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.8724 | 0.9383 | +0.0659 [+0.0625, +0.0692] | -0.0002, -0.0007 | 0.0075 | GAIN | GAIN |
| L-musique | webqsp | zero-shot | 0.3245 | 0.3229 | -0.0017 [-0.0190, +0.0152] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.7807 | 0.8040 | +0.0233 [+0.0204, +0.0261] | -0.0092, -0.0051 | 0.0149 | GAIN | GAIN |
| L-hotpotqa | squad | in-domain | 0.9109 | 0.9135 | +0.0026 [+0.0004, +0.0048] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5647 | 0.5754 | +0.0107 [+0.0030, +0.0182] | +0.0015, +0.0023 | 0.0075 | GAIN | GAIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8435 | 0.8499 | +0.0064 [+0.0017, +0.0110] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.8744 | 0.9333 | +0.0589 [+0.0555, +0.0623] | -0.0020, +0.0011 | 0.0075 | GAIN | GAIN |
| L-hotpotqa | webqsp | zero-shot | 0.2262 | 0.2680 | +0.0418 [+0.0228, +0.0595] | +0.0102, +0.0129 | 0.0233 | GAIN | GAIN |

Re-call: reads with a GAIN: 6 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> PROMISING; L-hotpotqa PROMISING -> PROMISING.
Changed calls: L-musique musique LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
