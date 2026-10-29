WQG (docs/WQG_ZG6_NULL_UNDER_WQ.md): G6RW's zg6r against WQK's seed-0 zg6 (beside there) under WQG's seed nulls (outputs/wqg/floors-<arm>; zspk's WQK's), beside, deciding nothing.

# wqg-zg6r-vs-zg6-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| WQ-musique | metaqa | zero-shot | -0.0288 [-0.0345, -0.0230] | +0.0139, -0.0020 | 0.0476 | LOSS | **WITHIN** |
| WQ-musique | squad | in-domain | +0.0008 [-0.0013, +0.0028] | +0.0003, +0.0023 | 0.0075 | WITHIN | WITHIN |
| WQ-musique | musique | zero-shot | -0.0912 [-0.1007, -0.0819] | +0.0502, +0.0512 | 0.0738 | LOSS | LOSS |
| WQ-musique | hotpotqa | in-domain | +0.0024 [+0.0003, +0.0045] | -0.0024, -0.0016 | 0.0075 | WITHIN | WITHIN |
| WQ-musique | 2wiki | in-domain | -0.0021 [-0.0037, -0.0005] | +0.0038, +0.0014 | 0.0075 | WITHIN | WITHIN |
| WQ-musique | webqsp | in-domain | +0.0078 [-0.0051, +0.0209] | -0.0074, -0.0026 | 0.0122 | WITHIN | WITHIN |
| WQ-hotpotqa | metaqa | zero-shot | -0.0333 [-0.0393, -0.0275] | +0.0521, +0.0322 | 0.0614 | LOSS | **WITHIN** |
| WQ-hotpotqa | squad | in-domain | -0.0003 [-0.0024, +0.0016] | +0.0019, +0.0013 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa | musique | in-domain | -0.0020 [-0.0075, +0.0034] | -0.0021, +0.0037 | 0.0080 | WITHIN | WITHIN |
| WQ-hotpotqa | hotpotqa | zero-shot | +0.0346 [+0.0309, +0.0386] | -0.0041, -0.0044 | 0.0099 | GAIN | GAIN |
| WQ-hotpotqa | 2wiki | in-domain | +0.0006 [-0.0010, +0.0023] | -0.0002, +0.0012 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa | webqsp | in-domain | +0.0116 [-0.0010, +0.0242] | -0.0053, -0.0135 | 0.0199 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 1 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: WQ-musique NO_GAIN -> NO_GAIN; WQ-hotpotqa MIXED -> PROMISING.
Changed calls: WQ-musique metaqa LOSS -> WITHIN; WQ-hotpotqa metaqa LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
