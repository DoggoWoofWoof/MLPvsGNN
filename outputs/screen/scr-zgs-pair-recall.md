# scr-zgs-pair re-called under the seed null: **PROMISING** (filed: MIXED)

R@5 of zgs's p@swa minus zret's p@swa of the same split (zret's fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2 and the thirteenth round). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zret R@5 | zgs R@5 | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.6496 | 0.6527 | +0.0030 [+0.0001, +0.0059] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9106 | 0.9116 | +0.0009 [-0.0011, +0.0029] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.3897 | 0.3798 | -0.0100 [-0.0165, -0.0032] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9061 | 0.9047 | -0.0014 [-0.0038, +0.0012] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.8714 | 0.8719 | +0.0005 [-0.0014, +0.0024] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.1766 | 0.1875 | +0.0109 [+0.0024, +0.0191] | -0.0070, -0.0018 | 0.0102 | GAIN | GAIN |
| L-hotpotqa | metaqa | in-domain | 0.6419 | 0.6497 | +0.0078 [+0.0047, +0.0109] | -0.0092, -0.0051 | 0.0149 | GAIN | **WITHIN** |
| L-hotpotqa | squad | in-domain | 0.9127 | 0.9117 | -0.0010 [-0.0032, +0.0011] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5623 | 0.5558 | -0.0065 [-0.0127, -0.0008] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8445 | 0.8392 | -0.0053 [-0.0087, -0.0016] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.8728 | 0.8736 | +0.0009 [-0.0012, +0.0030] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.1195 | 0.1227 | +0.0032 [-0.0046, +0.0117] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 1 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> PROMISING; L-hotpotqa PROMISING -> NO_GAIN.
Changed calls: L-musique musique LOSS -> WITHIN; L-hotpotqa metaqa GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
