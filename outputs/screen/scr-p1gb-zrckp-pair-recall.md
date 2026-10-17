P1g-b (docs/P1GB_POOLS_REFIT.md): zrckp on P1g-b's pools against zrckp on P1c2's.

# scr-p1gb-zrckp-pair re-called under the seed null: **PROMISING** (filed: MIXED)

R@5 of zrckp's p@swa minus zrckp's p@swa of the same split (zrckp's card fits on P1c2's pools), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/P1GB_POOLS_REFIT.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrckp R@5 (P1c2's pools) | zrckp R@5 (P1g-b's) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8019 | 0.7993 | -0.0027 [-0.0044, -0.0011] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9116 | 0.9127 | +0.0010 [-0.0009, +0.0030] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5121 | 0.5013 | -0.0108 [-0.0171, -0.0048] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9065 | 0.9035 | -0.0030 [-0.0057, -0.0005] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 | 0.9377 | +0.0030 [+0.0014, +0.0046] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 | 0.3768 | +0.0141 [+0.0015, +0.0257] | -0.0070, -0.0018 | 0.0102 | GAIN | GAIN |
| L-hotpotqa | metaqa | in-domain | 0.8044 | 0.7997 | -0.0048 [-0.0065, -0.0032] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 | 0.9122 | -0.0005 [-0.0024, +0.0013] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 | 0.5695 | -0.0050 [-0.0103, +0.0002] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 | 0.8546 | +0.0033 [-0.0001, +0.0068] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 | 0.9340 | -0.0010 [-0.0028, +0.0008] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 | 0.3364 | -0.0111 [-0.0245, +0.0025] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 1 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> PROMISING; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
