G6RW (docs/G6RW_G6R_UNDER_WQ.md): zg6r against WQK's seed-0 zspk under WQ on the T4, deciding.

# g6rw-zg6r-pair re-called under the seed null: **PROMISING** (filed: MIXED)

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| WQ-musique | metaqa | zero-shot | +0.0121 [+0.0058, +0.0184] | -0.0303, -0.0068 | 0.0439 | GAIN | **WITHIN** |
| WQ-musique | squad | in-domain | -0.0005 [-0.0025, +0.0014] | -0.0006, -0.0007 | 0.0075 | WITHIN | WITHIN |
| WQ-musique | musique | zero-shot | -0.0484 [-0.0562, -0.0410] | -0.0273, +0.0272 | 0.0545 | LOSS | **WITHIN** |
| WQ-musique | hotpotqa | in-domain | +0.0076 [+0.0053, +0.0097] | -0.0015, +0.0006 | 0.0075 | GAIN | GAIN |
| WQ-musique | 2wiki | in-domain | +0.0024 [+0.0009, +0.0038] | -0.0008, -0.0013 | 0.0075 | WITHIN | WITHIN |
| WQ-musique | webqsp | in-domain | +0.0167 [+0.0039, +0.0288] | -0.0123, -0.0007 | 0.0174 | GAIN | **WITHIN** |
| WQ-hotpotqa | metaqa | zero-shot | +0.0450 [+0.0380, +0.0519] | +0.0215, +0.0034 | 0.0308 | GAIN | GAIN |
| WQ-hotpotqa | squad | in-domain | -0.0003 [-0.0022, +0.0015] | -0.0007, +0.0000 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa | musique | in-domain | +0.0147 [+0.0086, +0.0207] | -0.0028, +0.0057 | 0.0090 | GAIN | GAIN |
| WQ-hotpotqa | hotpotqa | zero-shot | +0.0033 [-0.0001, +0.0068] | -0.0012, -0.0036 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa | 2wiki | in-domain | +0.0090 [+0.0074, +0.0107] | +0.0015, +0.0023 | 0.0075 | GAIN | GAIN |
| WQ-hotpotqa | webqsp | in-domain | +0.0383 [+0.0259, +0.0508] | +0.0020, +0.0060 | 0.0089 | GAIN | GAIN |

Re-call: reads with a GAIN: 5 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: WQ-musique MIXED -> PROMISING; WQ-hotpotqa PROMISING -> PROMISING.
Changed calls: WQ-musique metaqa GAIN -> WITHIN; WQ-musique musique LOSS -> WITHIN; WQ-musique webqsp GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
