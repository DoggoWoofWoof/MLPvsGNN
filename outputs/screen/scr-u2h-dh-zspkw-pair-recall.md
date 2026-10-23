U2h (docs/U2H_W2S_BESIDE.md), decision H (W2_S's KT columns beside today's graph, as an input to the lean pair): zspkw (KT_W beside KT, on U2d's base build) against zspk on U2d's base build (today's graph, P1c2's rule), re-called under U2d's zspk seed null.

# scr-u2h-dh-zspkw-pair re-called under the seed null: **NO_GAIN** (filed: PROMISING)

R@5 of the screen's p@swa minus the decision's base fit's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of U2d's own seeds 1 and 2 against the seed-0 fit on the base build) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | +0.0011 [+0.0003, +0.0019] | -0.0003, +0.0018 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | +0.0004 [-0.0013, +0.0023] | +0.0006, -0.0008 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | +0.0059 [+0.0003, +0.0118] | +0.0060, -0.0019 | 0.0089 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | -0.0015 [-0.0034, +0.0006] | -0.0030, -0.0003 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | +0.0023 [+0.0009, +0.0038] | -0.0003, +0.0012 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | +0.0030 [-0.0053, +0.0104] | -0.0071, -0.0022 | 0.0105 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | +0.0009 [+0.0001, +0.0018] | -0.0038, -0.0029 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | +0.0008 [-0.0011, +0.0028] | +0.0005, -0.0002 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | -0.0018 [-0.0064, +0.0033] | +0.0060, +0.0046 | 0.0107 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0059 [-0.0092, -0.0025] | -0.0018, +0.0101 | 0.0145 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | +0.0083 [+0.0066, +0.0102] | +0.0094, -0.0029 | 0.0139 | GAIN | **WITHIN** |
| L-hotpotqa | webqsp | zero-shot | -0.0030 [-0.0115, +0.0051] | -0.0275, +0.0079 | 0.0405 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa PROMISING -> NO_GAIN.
Changed calls: L-hotpotqa 2wiki GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
