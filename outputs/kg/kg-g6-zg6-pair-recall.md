# kg-g6-zg6-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | +0.0114 [+0.0097, +0.0131] | +0.0025, +0.0031 | 0.0075 | GAIN | GAIN |
| L-musique | squad | in-domain | +0.0018 [-0.0005, +0.0040] | +0.0007, +0.0012 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | -0.0009 [-0.0079, +0.0064] | -0.0066, -0.0072 | 0.0138 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | +0.0041 [+0.0017, +0.0065] | -0.0011, -0.0018 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | +0.0084 [+0.0067, +0.0100] | -0.0030, -0.0011 | 0.0075 | GAIN | GAIN |
| L-musique | webqsp | zero-shot | -0.0753 [-0.0957, -0.0527] | +0.0009, +0.0007 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | +0.0078 [+0.0063, +0.0093] | -0.0027, -0.0011 | 0.0075 | GAIN | GAIN |
| L-hotpotqa | squad | in-domain | -0.0017 [-0.0038, +0.0005] | +0.0001, -0.0001 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | +0.0199 [+0.0139, +0.0255] | +0.0008, +0.0074 | 0.0105 | GAIN | GAIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0159 [-0.0197, -0.0120] | -0.0047, +0.0001 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | 2wiki | in-domain | +0.0198 [+0.0177, +0.0218] | +0.0114, +0.0105 | 0.0219 | GAIN | **WITHIN** |
| L-hotpotqa | webqsp | zero-shot | -0.0210 [-0.0410, -0.0013] | -0.0166, +0.0041 | 0.0242 | LOSS | **WITHIN** |

Re-call: reads with a GAIN: 4 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> MIXED; L-hotpotqa MIXED -> MIXED.
Changed calls: L-hotpotqa 2wiki GAIN -> WITHIN; L-hotpotqa webqsp LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
