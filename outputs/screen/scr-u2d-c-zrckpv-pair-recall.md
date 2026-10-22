U2d (docs/U2D_REFIT_ON_U2E.md), decision C (the pools and via together): zrckpv on the new build against zrckp on the base build, re-called under U2d's zrckp seed null.

# scr-u2d-c-zrckpv-pair re-called under the seed null: **NO_GAIN** (filed: MIXED)

R@5 of the screen's p@swa minus the decision's base fit's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of U2d's own seeds 1 and 2 against the seed-0 fit on the base build) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | -0.0022 [-0.0037, -0.0007] | +0.0017, +0.0034 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | -0.0007 [-0.0029, +0.0017] | -0.0005, -0.0009 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | +0.0043 [-0.0031, +0.0113] | -0.0096, +0.0004 | 0.0136 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | -0.0605 [-0.0652, -0.0560] | -0.0005, +0.0011 | 0.0075 | LOSS | LOSS |
| L-musique | 2wiki | in-domain | -0.0198 [-0.0227, -0.0168] | -0.0020, +0.0017 | 0.0075 | LOSS | LOSS |
| L-musique | webqsp | zero-shot | +0.0168 [+0.0035, +0.0296] | +0.0035, +0.0124 | 0.0182 | GAIN | **WITHIN** |
| L-hotpotqa | metaqa | in-domain | -0.0031 [-0.0048, -0.0016] | -0.0008, -0.0010 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | +0.0004 [-0.0019, +0.0029] | -0.0001, -0.0003 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | +0.0040 [-0.0024, +0.0108] | +0.0084, +0.0013 | 0.0120 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0451 [-0.0499, -0.0406] | -0.0036, -0.0046 | 0.0083 | LOSS | LOSS |
| L-hotpotqa | 2wiki | in-domain | -0.0243 [-0.0272, -0.0212] | +0.0036, +0.0021 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | webqsp | zero-shot | +0.0150 [+0.0007, +0.0305] | -0.0158, +0.0164 | 0.0322 | GAIN | **WITHIN** |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 4. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> NO_GAIN; L-hotpotqa MIXED -> NO_GAIN.
Changed calls: L-musique webqsp GAIN -> WITHIN; L-hotpotqa webqsp GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
