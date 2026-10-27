U2n (docs/U2N_TYPED_CHANNELS.md): zg6e against KG's zspk on the T4, deciding.

# u2n-zg6e-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | +0.0117 [+0.0100, +0.0135] | +0.0025, +0.0031 | 0.0075 | GAIN | GAIN |
| L-musique | squad | in-domain | +0.0031 [+0.0009, +0.0054] | +0.0007, +0.0012 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | -0.0017 [-0.0088, +0.0052] | -0.0066, -0.0072 | 0.0138 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | +0.0034 [+0.0009, +0.0060] | -0.0011, -0.0018 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | +0.0074 [+0.0057, +0.0090] | -0.0030, -0.0011 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | -0.0988 [-0.1198, -0.0773] | +0.0009, +0.0007 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | +0.0076 [+0.0060, +0.0091] | -0.0027, -0.0011 | 0.0075 | GAIN | GAIN |
| L-hotpotqa | squad | in-domain | -0.0033 [-0.0055, -0.0011] | +0.0001, -0.0001 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | +0.0231 [+0.0169, +0.0291] | +0.0008, +0.0074 | 0.0105 | GAIN | GAIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0311 [-0.0352, -0.0267] | -0.0047, +0.0001 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | 2wiki | in-domain | +0.0172 [+0.0150, +0.0193] | +0.0114, +0.0105 | 0.0219 | GAIN | **WITHIN** |
| L-hotpotqa | webqsp | zero-shot | -0.0386 [-0.0603, -0.0170] | -0.0166, +0.0041 | 0.0242 | LOSS | LOSS |

Re-call: reads with a GAIN: 3 of 12; with a LOSS: 3. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> MIXED; L-hotpotqa MIXED -> MIXED.
Changed calls: L-hotpotqa 2wiki GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
