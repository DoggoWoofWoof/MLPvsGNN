# scr-zrm-pair re-called under the seed null: **PROMISING** (filed: PROMISING)

R@5 of zrm's p@swa minus zret's p@swa of the same split (zret's fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2 and the fourteenth round). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zret R@5 | zrm R@5 | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.6496 | 0.7789 | +0.1293 [+0.1241, +0.1347] | +0.0008, -0.0023 | 0.0075 | GAIN | GAIN |
| L-musique | squad | in-domain | 0.9106 | 0.9111 | +0.0004 [-0.0015, +0.0024] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.3897 | 0.5231 | +0.1334 [+0.1234, +0.1434] | +0.0503, -0.0079 | 0.0720 | GAIN | GAIN |
| L-musique | hotpotqa | in-domain | 0.9061 | 0.9040 | -0.0021 [-0.0044, +0.0003] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.8714 | 0.8709 | -0.0005 [-0.0023, +0.0014] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.1766 | 0.3128 | +0.1361 [+0.1127, +0.1597] | -0.0070, -0.0018 | 0.0102 | GAIN | GAIN |
| L-hotpotqa | metaqa | in-domain | 0.6419 | 0.7801 | +0.1381 [+0.1328, +0.1437] | -0.0092, -0.0051 | 0.0149 | GAIN | GAIN |
| L-hotpotqa | squad | in-domain | 0.9127 | 0.9116 | -0.0012 [-0.0032, +0.0008] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5623 | 0.5600 | -0.0023 [-0.0087, +0.0034] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8445 | 0.8457 | +0.0012 [-0.0024, +0.0049] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.8728 | 0.8681 | -0.0047 [-0.0068, -0.0025] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.1195 | 0.2416 | +0.1221 [+0.1018, +0.1424] | +0.0102, +0.0129 | 0.0233 | GAIN | GAIN |

Re-call: reads with a GAIN: 5 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique PROMISING -> PROMISING; L-hotpotqa PROMISING -> PROMISING.
Changed calls: none.
Within 0.00015 of the floor (the call stands): none.
