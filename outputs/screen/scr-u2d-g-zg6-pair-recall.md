U2d (docs/U2D_REFIT_ON_U2E.md), decision G (the pools, for zg6): zg6 on the new build against zg6 on the base build, re-called under U2d's zspk seed null re-based through scr-u2d-base-zg6-vs-zspk, scr-u2d-base-zg6-vs-zspk-hp.

# scr-u2d-g-zg6-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of the screen's p@swa minus the decision's base fit's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of U2d's own seeds 1 and 2 against the seed-0 fit on the base build) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | +0.0011 [-0.0002, +0.0023] | -0.0003, +0.0018 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | +0.0035 [+0.0013, +0.0057] | +0.0006, -0.0008 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | +0.0220 [+0.0153, +0.0287] | +0.0060, -0.0019 | 0.0089 | GAIN | GAIN |
| L-musique | hotpotqa | in-domain | -0.0514 [-0.0559, -0.0467] | -0.0030, -0.0003 | 0.0075 | LOSS | LOSS |
| L-musique | 2wiki | in-domain | -0.0110 [-0.0137, -0.0082] | -0.0003, +0.0012 | 0.0075 | LOSS | LOSS |
| L-musique | webqsp | zero-shot | -0.0808 [-0.0980, -0.0632] | -0.0071, -0.0022 | 0.0105 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | +0.0022 [+0.0010, +0.0034] | -0.0038, -0.0029 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | +0.0019 [-0.0004, +0.0040] | +0.0005, -0.0002 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | +0.0048 [-0.0008, +0.0105] | +0.0060, +0.0046 | 0.0107 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0499 [-0.0545, -0.0450] | -0.0018, +0.0101 | 0.0145 | LOSS | LOSS |
| L-hotpotqa | 2wiki | in-domain | -0.0124 [-0.0152, -0.0098] | +0.0094, -0.0029 | 0.0139 | LOSS | **WITHIN** |
| L-hotpotqa | webqsp | zero-shot | -0.0408 [-0.0566, -0.0243] | -0.0275, +0.0079 | 0.0405 | LOSS | LOSS |

Re-call: reads with a GAIN: 1 of 12; with a LOSS: 5. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> MIXED; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-hotpotqa 2wiki LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
