# wqk-g6-zg6-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| WQ-musique | metaqa | zero-shot | +0.0409 [+0.0348, +0.0464] | -0.0303, -0.0068 | 0.0439 | GAIN | **WITHIN** |
| WQ-musique | squad | in-domain | -0.0013 [-0.0032, +0.0008] | -0.0006, -0.0007 | 0.0075 | WITHIN | WITHIN |
| WQ-musique | musique | zero-shot | +0.0428 [+0.0336, +0.0518] | -0.0273, +0.0272 | 0.0545 | GAIN | **WITHIN** |
| WQ-musique | hotpotqa | in-domain | +0.0051 [+0.0028, +0.0076] | -0.0015, +0.0006 | 0.0075 | WITHIN | WITHIN |
| WQ-musique | 2wiki | in-domain | +0.0045 [+0.0030, +0.0061] | -0.0008, -0.0013 | 0.0075 | WITHIN | WITHIN |
| WQ-musique | webqsp | in-domain | +0.0089 [-0.0038, +0.0215] | -0.0123, -0.0007 | 0.0174 | WITHIN | WITHIN |
| WQ-hotpotqa | metaqa | zero-shot | +0.0783 [+0.0713, +0.0853] | +0.0215, +0.0034 | 0.0308 | GAIN | GAIN |
| WQ-hotpotqa | squad | in-domain | +0.0000 [-0.0019, +0.0019] | -0.0007, +0.0000 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa | musique | in-domain | +0.0167 [+0.0109, +0.0230] | -0.0028, +0.0057 | 0.0090 | GAIN | GAIN |
| WQ-hotpotqa | hotpotqa | zero-shot | -0.0313 [-0.0354, -0.0269] | -0.0012, -0.0036 | 0.0075 | LOSS | LOSS |
| WQ-hotpotqa | 2wiki | in-domain | +0.0084 [+0.0065, +0.0102] | +0.0015, +0.0023 | 0.0075 | GAIN | GAIN |
| WQ-hotpotqa | webqsp | in-domain | +0.0268 [+0.0149, +0.0387] | +0.0020, +0.0060 | 0.0089 | GAIN | GAIN |

Re-call: reads with a GAIN: 4 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: WQ-musique PROMISING -> NO_GAIN; WQ-hotpotqa MIXED -> MIXED.
Changed calls: WQ-musique metaqa GAIN -> WITHIN; WQ-musique musique GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
