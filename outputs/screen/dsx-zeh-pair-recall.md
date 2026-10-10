# dsx-zeh-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of zeh's p@swa minus zrc's p@swa of the same split (zrc's screen fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2 and round 33). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrc R@5 | new R@5 | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7782 | 0.7766 | -0.0016 [-0.0029, -0.0003] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9105 | 0.9089 | -0.0016 [-0.0040, +0.0006] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5235 | 0.5129 | -0.0107 [-0.0157, -0.0058] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9041 | 0.9003 | -0.0038 [-0.0063, -0.0011] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.8733 | 0.8702 | -0.0031 [-0.0051, -0.0011] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3384 | 0.3282 | -0.0102 [-0.0197, -0.0002] | -0.0070, -0.0018 | 0.0102 | LOSS | **WITHIN** |
| L-hotpotqa | metaqa | in-domain | 0.7793 | 0.7757 | -0.0036 [-0.0050, -0.0023] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9112 | 0.9068 | -0.0045 [-0.0072, -0.0018] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5645 | 0.5656 | +0.0012 [-0.0052, +0.0078] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8451 | 0.8435 | -0.0016 [-0.0055, +0.0020] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.8731 | 0.8642 | -0.0089 [-0.0112, -0.0066] | -0.0020, +0.0011 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | webqsp | zero-shot | 0.2865 | 0.2751 | -0.0115 [-0.0215, -0.0016] | +0.0102, +0.0129 | 0.0233 | LOSS | **WITHIN** |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique LOSS -> WITHIN; L-musique webqsp LOSS -> WITHIN; L-hotpotqa webqsp LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): L-musique webqsp (-0.0102 against 0.0102).
