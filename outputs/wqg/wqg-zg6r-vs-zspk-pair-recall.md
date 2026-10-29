WQG (docs/WQG_ZG6_NULL_UNDER_WQ.md): G6RW's zg6r against WQK's seed-0 zspk (the deciding comparison there) under WQG's seed nulls (outputs/wqg/floors-<arm>; zspk's WQK's), beside, deciding nothing.

# wqg-zg6r-vs-zspk-pair re-called under the seed null: **PROMISING** (filed: MIXED)

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| WQ-musique | metaqa | zero-shot | +0.0121 [+0.0058, +0.0184] | +0.0139, -0.0020 | 0.0341 | GAIN | **WITHIN** |
| WQ-musique | squad | in-domain | -0.0005 [-0.0025, +0.0014] | +0.0003, +0.0023 | 0.0075 | WITHIN | WITHIN |
| WQ-musique | musique | zero-shot | -0.0484 [-0.0562, -0.0410] | +0.0502, +0.0512 | 0.0814 | LOSS | **WITHIN** |
| WQ-musique | hotpotqa | in-domain | +0.0076 [+0.0053, +0.0097] | -0.0024, -0.0016 | 0.0075 | GAIN | GAIN |
| WQ-musique | 2wiki | in-domain | +0.0024 [+0.0009, +0.0038] | +0.0038, +0.0014 | 0.0075 | WITHIN | WITHIN |
| WQ-musique | webqsp | in-domain | +0.0167 [+0.0039, +0.0288] | -0.0074, -0.0026 | 0.0146 | GAIN | GAIN |
| WQ-hotpotqa | metaqa | zero-shot | +0.0450 [+0.0380, +0.0519] | +0.0521, +0.0322 | 0.0650 | GAIN | **WITHIN** |
| WQ-hotpotqa | squad | in-domain | -0.0003 [-0.0022, +0.0015] | +0.0019, +0.0013 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa | musique | in-domain | +0.0147 [+0.0086, +0.0207] | -0.0021, +0.0037 | 0.0076 | GAIN | GAIN |
| WQ-hotpotqa | hotpotqa | zero-shot | +0.0033 [-0.0001, +0.0068] | -0.0041, -0.0044 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa | 2wiki | in-domain | +0.0090 [+0.0074, +0.0107] | -0.0002, +0.0012 | 0.0075 | GAIN | GAIN |
| WQ-hotpotqa | webqsp | in-domain | +0.0383 [+0.0259, +0.0508] | -0.0053, -0.0135 | 0.0158 | GAIN | GAIN |

Re-call: reads with a GAIN: 5 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: WQ-musique MIXED -> PROMISING; WQ-hotpotqa PROMISING -> PROMISING.
Changed calls: WQ-musique metaqa GAIN -> WITHIN; WQ-musique musique LOSS -> WITHIN; WQ-hotpotqa metaqa GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
