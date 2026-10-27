G6b2 (docs/G6B2_ROW_BACKOFF.md): zg6b2 against KG's zspk on the T4, deciding.

# g6b2-zg6b2-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | +0.0099 [+0.0083, +0.0115] | +0.0025, +0.0031 | 0.0075 | GAIN | GAIN |
| L-musique | squad | in-domain | +0.0017 [-0.0007, +0.0040] | +0.0007, +0.0012 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | -0.0366 [-0.0446, -0.0289] | -0.0066, -0.0072 | 0.0138 | LOSS | LOSS |
| L-musique | hotpotqa | in-domain | +0.0027 [+0.0004, +0.0051] | -0.0011, -0.0018 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | +0.0067 [+0.0051, +0.0083] | -0.0030, -0.0011 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | -0.0929 [-0.1134, -0.0726] | +0.0009, +0.0007 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | +0.0079 [+0.0063, +0.0094] | -0.0027, -0.0011 | 0.0075 | GAIN | GAIN |
| L-hotpotqa | squad | in-domain | -0.0012 [-0.0032, +0.0008] | +0.0001, -0.0001 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | +0.0248 [+0.0189, +0.0305] | +0.0008, +0.0074 | 0.0105 | GAIN | GAIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0182 [-0.0223, -0.0142] | -0.0047, +0.0001 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | 2wiki | in-domain | +0.0154 [+0.0134, +0.0175] | +0.0114, +0.0105 | 0.0219 | GAIN | **WITHIN** |
| L-hotpotqa | webqsp | zero-shot | -0.0139 [-0.0343, +0.0058] | -0.0166, +0.0041 | 0.0242 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 3 of 12; with a LOSS: 3. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> MIXED; L-hotpotqa MIXED -> MIXED.
Changed calls: L-hotpotqa 2wiki GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
