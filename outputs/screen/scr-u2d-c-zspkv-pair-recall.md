U2d (docs/U2D_REFIT_ON_U2E.md), decision C (the pools and via together): zspkv on the new build against zspk on the base build, re-called under U2d's zspk seed null.

# scr-u2d-c-zspkv-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of the screen's p@swa minus the decision's base fit's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of U2d's own seeds 1 and 2 against the seed-0 fit on the base build) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | -0.0024 [-0.0040, -0.0010] | -0.0003, +0.0018 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | +0.0009 [-0.0014, +0.0033] | +0.0006, -0.0008 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | +0.0121 [+0.0049, +0.0194] | +0.0060, -0.0019 | 0.0089 | GAIN | GAIN |
| L-musique | hotpotqa | in-domain | -0.0625 [-0.0671, -0.0577] | -0.0030, -0.0003 | 0.0075 | LOSS | LOSS |
| L-musique | 2wiki | in-domain | -0.0187 [-0.0216, -0.0157] | -0.0003, +0.0012 | 0.0075 | LOSS | LOSS |
| L-musique | webqsp | zero-shot | -0.0012 [-0.0139, +0.0113] | -0.0071, -0.0022 | 0.0105 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | -0.0053 [-0.0069, -0.0037] | -0.0038, -0.0029 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | +0.0026 [+0.0003, +0.0049] | +0.0005, -0.0002 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | -0.0072 [-0.0132, -0.0006] | +0.0060, +0.0046 | 0.0107 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0427 [-0.0473, -0.0384] | -0.0018, +0.0101 | 0.0145 | LOSS | LOSS |
| L-hotpotqa | 2wiki | in-domain | -0.0183 [-0.0213, -0.0152] | +0.0094, -0.0029 | 0.0139 | LOSS | LOSS |
| L-hotpotqa | webqsp | zero-shot | +0.0094 [-0.0042, +0.0230] | -0.0275, +0.0079 | 0.0405 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 1 of 12; with a LOSS: 4. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> MIXED; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: none.
Within 0.00015 of the floor (the call stands): none.
