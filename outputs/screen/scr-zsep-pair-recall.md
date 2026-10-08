# scr-zsep-pair re-called under the seed null: **NO_GAIN** (filed: MIXED)

R@5 of zsep's p@swa minus zret's p@swa of the same split (zret's fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2 and the sixteenth round). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zret R@5 | zsep R@5 | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.6496 | 0.6540 | +0.0043 [+0.0016, +0.0073] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9106 | 0.9095 | -0.0012 [-0.0031, +0.0008] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.3897 | 0.3210 | -0.0687 [-0.0765, -0.0610] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9061 | 0.9031 | -0.0030 [-0.0054, -0.0004] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.8714 | 0.8708 | -0.0007 [-0.0025, +0.0013] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.1766 | 0.1770 | +0.0003 [-0.0095, +0.0096] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.6419 | 0.6524 | +0.0104 [+0.0073, +0.0135] | -0.0092, -0.0051 | 0.0149 | GAIN | **WITHIN** |
| L-hotpotqa | squad | in-domain | 0.9127 | 0.9112 | -0.0015 [-0.0037, +0.0007] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5623 | 0.5522 | -0.0101 [-0.0163, -0.0043] | +0.0015, +0.0023 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | hotpotqa | zero-shot | 0.8445 | 0.8419 | -0.0026 [-0.0061, +0.0009] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.8728 | 0.8725 | -0.0003 [-0.0024, +0.0018] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.1195 | 0.1189 | -0.0006 [-0.0105, +0.0088] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa MIXED -> NO_GAIN.
Changed calls: L-musique musique LOSS -> WITHIN; L-hotpotqa metaqa GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
