P1g-b (docs/P1GB_POOLS_REFIT.md): zspk on P1g-b's pools against zspk on P1c2's.

# scr-p1gb-zspk-pair re-called under the seed null: **NO_GAIN** (filed: MIXED)

R@5 of zspk's p@swa minus zspk's p@swa of the same split (zspk's card fits on P1c2's pools), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/P1GB_POOLS_REFIT.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zspk R@5 (P1c2's pools) | zspk R@5 (P1g-b's) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8009 | 0.8004 | -0.0005 [-0.0024, +0.0013] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9114 | 0.9129 | +0.0015 [-0.0004, +0.0035] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5012 | 0.5169 | +0.0157 [+0.0098, +0.0221] | +0.0503, -0.0079 | 0.0720 | GAIN | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9080 | 0.9086 | +0.0006 [-0.0020, +0.0033] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9383 | 0.9344 | -0.0038 [-0.0053, -0.0024] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3229 | 0.3354 | +0.0125 [-0.0010, +0.0253] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8040 | 0.8011 | -0.0029 [-0.0045, -0.0013] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9135 | 0.9127 | -0.0008 [-0.0029, +0.0013] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5754 | 0.5662 | -0.0092 [-0.0145, -0.0039] | +0.0015, +0.0023 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | hotpotqa | zero-shot | 0.8499 | 0.8483 | -0.0016 [-0.0047, +0.0016] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9333 | 0.9326 | -0.0007 [-0.0023, +0.0010] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2680 | 0.2808 | +0.0129 [-0.0003, +0.0274] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique PROMISING -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
