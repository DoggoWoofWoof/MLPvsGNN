U2d (docs/U2D_REFIT_ON_U2E.md), decision A (the pools, for the lean pair): zrckp on the new build against zrckp on the base build, re-called under U2d's zrckp seed null.

# scr-u2d-a-zrckp-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of the screen's p@swa minus the decision's base fit's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of U2d's own seeds 1 and 2 against the seed-0 fit on the base build) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | -0.0009 [-0.0024, +0.0006] | +0.0017, +0.0034 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | +0.0008 [-0.0014, +0.0030] | -0.0005, -0.0009 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | +0.0036 [-0.0037, +0.0104] | -0.0096, +0.0004 | 0.0136 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | -0.0571 [-0.0617, -0.0527] | -0.0005, +0.0011 | 0.0075 | LOSS | LOSS |
| L-musique | 2wiki | in-domain | -0.0141 [-0.0169, -0.0112] | -0.0020, +0.0017 | 0.0075 | LOSS | LOSS |
| L-musique | webqsp | zero-shot | +0.0390 [+0.0260, +0.0523] | +0.0035, +0.0124 | 0.0182 | GAIN | GAIN |
| L-hotpotqa | metaqa | in-domain | -0.0023 [-0.0039, -0.0009] | -0.0008, -0.0010 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | +0.0007 [-0.0015, +0.0029] | -0.0001, -0.0003 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | +0.0044 [-0.0020, +0.0109] | +0.0084, +0.0013 | 0.0120 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0474 [-0.0521, -0.0428] | -0.0036, -0.0046 | 0.0083 | LOSS | LOSS |
| L-hotpotqa | 2wiki | in-domain | -0.0273 [-0.0304, -0.0245] | +0.0036, +0.0021 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | webqsp | zero-shot | +0.0500 [+0.0362, +0.0644] | -0.0158, +0.0164 | 0.0322 | GAIN | GAIN |

Re-call: reads with a GAIN: 2 of 12; with a LOSS: 4. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> MIXED; L-hotpotqa MIXED -> MIXED.
Changed calls: none.
Within 0.00015 of the floor (the call stands): none.
