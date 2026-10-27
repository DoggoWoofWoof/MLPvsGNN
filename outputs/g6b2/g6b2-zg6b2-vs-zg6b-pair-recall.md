G6b2 (docs/G6B2_ROW_BACKOFF.md): zg6b2 against G6b's zg6b on the T4, beside, deciding nothing.

# g6b2-zg6b2-vs-zg6b-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | -0.0014 [-0.0023, -0.0006] | +0.0025, +0.0031 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | +0.0006 [-0.0012, +0.0024] | +0.0007, +0.0012 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | -0.0272 [-0.0327, -0.0220] | -0.0066, -0.0072 | 0.0138 | LOSS | LOSS |
| L-musique | hotpotqa | in-domain | -0.0015 [-0.0036, +0.0005] | -0.0011, -0.0018 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | +0.0007 [-0.0006, +0.0020] | -0.0030, -0.0011 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | -0.0562 [-0.0706, -0.0420] | +0.0009, +0.0007 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | +0.0001 [-0.0008, +0.0010] | -0.0027, -0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | -0.0003 [-0.0019, +0.0013] | +0.0001, -0.0001 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | +0.0068 [+0.0020, +0.0115] | +0.0008, +0.0074 | 0.0105 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0020 [-0.0048, +0.0009] | -0.0047, +0.0001 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | -0.0023 [-0.0037, -0.0009] | +0.0114, +0.0105 | 0.0219 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | +0.0104 [-0.0019, +0.0237] | -0.0166, +0.0041 | 0.0242 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: none.
Within 0.00015 of the floor (the call stands): none.
