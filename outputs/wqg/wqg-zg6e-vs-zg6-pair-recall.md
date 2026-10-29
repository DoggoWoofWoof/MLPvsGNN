WQG (docs/WQG_ZG6_NULL_UNDER_WQ.md): R2EG's zg6e against WQK's seed-0 zg6 under WQG's seed nulls (outputs/wqg/floors-<arm>; zspk's WQK's), beside, deciding nothing.

# wqg-zg6e-vs-zg6-pair re-called under the seed null: **NO_GAIN** (filed: MIXED)

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| WQ-musique | metaqa | zero-shot | -0.0443 [-0.0505, -0.0383] | -0.0154, -0.0428 | 0.0643 | LOSS | **WITHIN** |
| WQ-musique | squad | in-domain | +0.0004 [-0.0018, +0.0026] | +0.0016, +0.0027 | 0.0075 | WITHIN | WITHIN |
| WQ-musique | musique | zero-shot | -0.0180 [-0.0250, -0.0104] | +0.0150, +0.0093 | 0.0250 | LOSS | **WITHIN** |
| WQ-musique | hotpotqa | in-domain | +0.0036 [+0.0013, +0.0058] | +0.0015, +0.0016 | 0.0075 | WITHIN | WITHIN |
| WQ-musique | 2wiki | in-domain | -0.0010 [-0.0027, +0.0006] | +0.0012, +0.0024 | 0.0075 | WITHIN | WITHIN |
| WQ-musique | webqsp | in-domain | +0.0015 [-0.0106, +0.0149] | -0.0012, -0.0093 | 0.0133 | WITHIN | WITHIN |
| WQ-hotpotqa | metaqa | zero-shot | -0.0320 [-0.0374, -0.0266] | -0.0037, +0.0025 | 0.0075 | LOSS | LOSS |
| WQ-hotpotqa | squad | in-domain | -0.0006 [-0.0029, +0.0018] | -0.0004, +0.0003 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa | musique | in-domain | -0.0033 [-0.0092, +0.0024] | -0.0065, -0.0017 | 0.0095 | WITHIN | WITHIN |
| WQ-hotpotqa | hotpotqa | zero-shot | +0.0097 [+0.0059, +0.0134] | +0.0078, +0.0012 | 0.0112 | GAIN | **WITHIN** |
| WQ-hotpotqa | 2wiki | in-domain | -0.0015 [-0.0032, +0.0002] | -0.0017, +0.0007 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa | webqsp | in-domain | -0.0078 [-0.0203, +0.0043] | -0.0081, -0.0110 | 0.0193 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: WQ-musique NO_GAIN -> NO_GAIN; WQ-hotpotqa MIXED -> NO_GAIN.
Changed calls: WQ-musique metaqa LOSS -> WITHIN; WQ-musique musique LOSS -> WITHIN; WQ-hotpotqa hotpotqa GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
