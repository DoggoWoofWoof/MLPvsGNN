U2n (docs/U2N_TYPED_CHANNELS.md): zg6e against KG's zg6 on the T4, beside, deciding nothing.

# u2n-zg6e-vs-zg6-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | +0.0003 [-0.0005, +0.0011] | +0.0025, +0.0031 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | +0.0013 [-0.0005, +0.0032] | +0.0007, +0.0012 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | -0.0008 [-0.0054, +0.0041] | -0.0066, -0.0072 | 0.0138 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | -0.0007 [-0.0026, +0.0014] | -0.0011, -0.0018 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | -0.0010 [-0.0023, +0.0003] | -0.0030, -0.0011 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | -0.0235 [-0.0372, -0.0108] | +0.0009, +0.0007 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | -0.0002 [-0.0011, +0.0007] | -0.0027, -0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | -0.0016 [-0.0035, +0.0003] | +0.0001, -0.0001 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | +0.0032 [-0.0016, +0.0081] | +0.0008, +0.0074 | 0.0105 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0153 [-0.0184, -0.0120] | -0.0047, +0.0001 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | 2wiki | in-domain | -0.0026 [-0.0041, -0.0012] | +0.0114, +0.0105 | 0.0219 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | -0.0176 [-0.0324, -0.0044] | -0.0166, +0.0041 | 0.0242 | LOSS | **WITHIN** |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-hotpotqa webqsp LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
