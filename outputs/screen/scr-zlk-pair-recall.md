# scr-zlk-pair re-called under the seed null: **PROMISING** (filed: PROMISING)

R@5 of zlk's p@swa minus zrm's p@swa of the same split (zrm's screen fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2 and the twenty-second round). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrm R@5 | zlk R@5 | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7789 | 0.7746 | -0.0044 [-0.0058, -0.0030] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9111 | 0.9104 | -0.0007 [-0.0026, +0.0013] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5231 | 0.5204 | -0.0027 [-0.0088, +0.0030] | +0.0503, -0.0079 | 0.0720 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | 0.9040 | 0.9041 | +0.0001 [-0.0024, +0.0026] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.8709 | 0.8668 | -0.0041 [-0.0061, -0.0019] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3128 | 0.3211 | +0.0083 [-0.0007, +0.0186] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.7801 | 0.7787 | -0.0014 [-0.0027, -0.0001] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9116 | 0.9135 | +0.0019 [-0.0003, +0.0040] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5600 | 0.5684 | +0.0084 [+0.0027, +0.0142] | +0.0015, +0.0023 | 0.0075 | GAIN | GAIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8457 | 0.8468 | +0.0011 [-0.0026, +0.0045] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.8681 | 0.8728 | +0.0047 [+0.0025, +0.0069] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2416 | 0.2471 | +0.0055 [-0.0048, +0.0168] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 1 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa PROMISING -> PROMISING.
Changed calls: none.
Within 0.00015 of the floor (the call stands): none.
