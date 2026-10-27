A5cp (docs/A5CP_P1GB_POOLS.md): zrckpc (zrckp with zck's coupled kind offsets on typed graphs (zret's output layer and base weight, trained jointly)) against zrckp, both on P1g-b's pools.

# scr-a5cp-zrckpc-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of zrckpc's p@swa minus zrckp's p@swa of the same split (zrckp's card fits on P1g-b's pools), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/A5CP_P1GB_POOLS.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrckp R@5 (P1g-b's) | zrckpc R@5 (A5c's offsets, P1g-b's) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7993 | 0.7997 | +0.0004 [-0.0006, +0.0015] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9127 | 0.9117 | -0.0009 [-0.0027, +0.0009] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5013 | 0.5111 | +0.0098 [+0.0051, +0.0147] | +0.0503, -0.0079 | 0.0720 | GAIN | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9035 | 0.9061 | +0.0026 [+0.0003, +0.0050] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9377 | 0.9332 | -0.0045 [-0.0060, -0.0030] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3768 | 0.3928 | +0.0160 [+0.0039, +0.0280] | -0.0070, -0.0018 | 0.0102 | GAIN | GAIN |
| L-hotpotqa | metaqa | in-domain | 0.7997 | 0.8019 | +0.0022 [+0.0012, +0.0034] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9122 | 0.9122 | +0.0000 [-0.0019, +0.0019] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5695 | 0.5695 | +0.0000 [-0.0053, +0.0050] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8546 | 0.8527 | -0.0019 [-0.0051, +0.0011] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9340 | 0.9256 | -0.0084 [-0.0100, -0.0066] | -0.0020, +0.0011 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | webqsp | zero-shot | 0.3364 | 0.3730 | +0.0366 [+0.0216, +0.0522] | +0.0102, +0.0129 | 0.0233 | GAIN | GAIN |

Re-call: reads with a GAIN: 2 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique PROMISING -> PROMISING; L-hotpotqa MIXED -> MIXED.
Changed calls: L-musique musique GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
