G6b (docs/G6B_GATED_BACKOFF.md): zg6b against KG's zspk on the T4, deciding.

# g6b-zg6b-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | +0.0114 [+0.0097, +0.0130] | +0.0025, +0.0031 | 0.0075 | GAIN | GAIN |
| L-musique | squad | in-domain | +0.0011 [-0.0012, +0.0033] | +0.0007, +0.0012 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | -0.0094 [-0.0166, -0.0022] | -0.0066, -0.0072 | 0.0138 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | +0.0042 [+0.0019, +0.0065] | -0.0011, -0.0018 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | +0.0060 [+0.0044, +0.0075] | -0.0030, -0.0011 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | -0.0367 [-0.0571, -0.0172] | +0.0009, +0.0007 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | +0.0078 [+0.0061, +0.0094] | -0.0027, -0.0011 | 0.0075 | GAIN | GAIN |
| L-hotpotqa | squad | in-domain | -0.0009 [-0.0031, +0.0012] | +0.0001, -0.0001 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | +0.0180 [+0.0119, +0.0243] | +0.0008, +0.0074 | 0.0105 | GAIN | GAIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0162 [-0.0201, -0.0124] | -0.0047, +0.0001 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | 2wiki | in-domain | +0.0177 [+0.0157, +0.0197] | +0.0114, +0.0105 | 0.0219 | GAIN | **WITHIN** |
| L-hotpotqa | webqsp | zero-shot | -0.0243 [-0.0439, -0.0043] | -0.0166, +0.0041 | 0.0242 | LOSS | LOSS |

Re-call: reads with a GAIN: 3 of 12; with a LOSS: 3. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> MIXED; L-hotpotqa MIXED -> MIXED.
Changed calls: L-musique musique LOSS -> WITHIN; L-hotpotqa 2wiki GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): L-hotpotqa webqsp (-0.0243 against 0.0242).
