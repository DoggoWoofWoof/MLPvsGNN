G6RW (docs/G6RW_G6R_UNDER_WQ.md): zg6r against WQK's seed-0 zg6 under WQ on the T4, beside, deciding nothing.

# g6rw-zg6r-vs-zg6-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| WQ-musique | metaqa | zero-shot | -0.0288 [-0.0345, -0.0230] | -0.0303, -0.0068 | 0.0439 | LOSS | **WITHIN** |
| WQ-musique | squad | in-domain | +0.0008 [-0.0013, +0.0028] | -0.0006, -0.0007 | 0.0075 | WITHIN | WITHIN |
| WQ-musique | musique | zero-shot | -0.0912 [-0.1007, -0.0819] | -0.0273, +0.0272 | 0.0545 | LOSS | LOSS |
| WQ-musique | hotpotqa | in-domain | +0.0024 [+0.0003, +0.0045] | -0.0015, +0.0006 | 0.0075 | WITHIN | WITHIN |
| WQ-musique | 2wiki | in-domain | -0.0021 [-0.0037, -0.0005] | -0.0008, -0.0013 | 0.0075 | WITHIN | WITHIN |
| WQ-musique | webqsp | in-domain | +0.0078 [-0.0051, +0.0209] | -0.0123, -0.0007 | 0.0174 | WITHIN | WITHIN |
| WQ-hotpotqa | metaqa | zero-shot | -0.0333 [-0.0393, -0.0275] | +0.0215, +0.0034 | 0.0308 | LOSS | LOSS |
| WQ-hotpotqa | squad | in-domain | -0.0003 [-0.0024, +0.0016] | -0.0007, +0.0000 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa | musique | in-domain | -0.0020 [-0.0075, +0.0034] | -0.0028, +0.0057 | 0.0090 | WITHIN | WITHIN |
| WQ-hotpotqa | hotpotqa | zero-shot | +0.0346 [+0.0309, +0.0386] | -0.0012, -0.0036 | 0.0075 | GAIN | GAIN |
| WQ-hotpotqa | 2wiki | in-domain | +0.0006 [-0.0010, +0.0023] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa | webqsp | in-domain | +0.0116 [-0.0010, +0.0242] | +0.0020, +0.0060 | 0.0089 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 1 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: WQ-musique NO_GAIN -> NO_GAIN; WQ-hotpotqa MIXED -> MIXED.
Changed calls: WQ-musique metaqa LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
