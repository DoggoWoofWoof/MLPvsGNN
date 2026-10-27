A7p (docs/A7P_P1GB_POOLS.md): zrckps (zrckp plus the A7 head) against zrckp, both on P1g-b's pools.

# scr-a7p-zrckps-pair re-called under the seed null: **PROMISING** (filed: PROMISING)

R@5 of zrckps's p@swa minus zrckp's p@swa of the same split (zrckp's card fits on P1g-b's pools), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/A7P_P1GB_POOLS.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrckp R@5 (P1g-b's) | zrckps R@5 (A7's scorer input, P1g-b's) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7993 | 0.8014 | +0.0022 [+0.0011, +0.0033] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9127 | 0.9126 | -0.0001 [-0.0019, +0.0016] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5013 | 0.5127 | +0.0114 [+0.0065, +0.0164] | +0.0503, -0.0079 | 0.0720 | GAIN | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9035 | 0.9086 | +0.0051 [+0.0030, +0.0074] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9377 | 0.9373 | -0.0004 [-0.0017, +0.0011] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3768 | 0.3962 | +0.0194 [+0.0092, +0.0296] | -0.0070, -0.0018 | 0.0102 | GAIN | GAIN |
| L-hotpotqa | metaqa | in-domain | 0.7997 | 0.8041 | +0.0044 [+0.0033, +0.0056] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9122 | 0.9139 | +0.0017 [-0.0003, +0.0035] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5695 | 0.5698 | +0.0003 [-0.0050, +0.0060] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8546 | 0.8623 | +0.0076 [+0.0045, +0.0107] | -0.0055, -0.0130 | 0.0200 | GAIN | **WITHIN** |
| L-hotpotqa | 2wiki | in-domain | 0.9340 | 0.9333 | -0.0007 [-0.0022, +0.0009] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3364 | 0.3637 | +0.0273 [+0.0147, +0.0402] | +0.0102, +0.0129 | 0.0233 | GAIN | GAIN |

Re-call: reads with a GAIN: 2 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique PROMISING -> PROMISING; L-hotpotqa PROMISING -> PROMISING.
Changed calls: L-musique musique GAIN -> WITHIN; L-hotpotqa hotpotqa GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
