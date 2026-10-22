U2d (docs/U2D_REFIT_ON_U2E.md), decision B (via, on the new pools): zrckpv on the new build against zrckp on the new build, re-called under U2d's zrckp seed null re-based through scr-u2d-a-zrckp, scr-u2d-a-zrckp-hp.

# scr-u2d-b-zrckpv-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of the screen's p@swa minus the decision's base fit's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of U2d's own seeds 1 and 2 against the seed-0 fit on the base build) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | -0.0013 [-0.0022, -0.0004] | +0.0017, +0.0034 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | -0.0015 [-0.0034, +0.0003] | -0.0005, -0.0009 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | +0.0007 [-0.0040, +0.0056] | -0.0096, +0.0004 | 0.0136 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | -0.0034 [-0.0059, -0.0010] | -0.0005, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | -0.0057 [-0.0075, -0.0042] | -0.0020, +0.0017 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | -0.0222 [-0.0327, -0.0117] | +0.0035, +0.0124 | 0.0182 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | -0.0007 [-0.0018, +0.0003] | -0.0008, -0.0010 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | -0.0003 [-0.0019, +0.0015] | -0.0001, -0.0003 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | -0.0003 [-0.0059, +0.0049] | +0.0084, +0.0013 | 0.0120 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | +0.0023 [-0.0009, +0.0055] | -0.0036, -0.0046 | 0.0083 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | +0.0030 [+0.0010, +0.0052] | +0.0036, +0.0021 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | -0.0350 [-0.0478, -0.0217] | -0.0158, +0.0164 | 0.0322 | LOSS | LOSS |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: none.
Within 0.00015 of the floor (the call stands): none.
