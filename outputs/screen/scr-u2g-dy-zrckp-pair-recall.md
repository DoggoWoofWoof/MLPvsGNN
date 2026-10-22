U2g (docs/U2G_GRAPH_AND_RULE.md), decision Y (W2_S at P1c2's rule, for the lean pair): zrckp on Y (W2_S, P1c2's rule) against zrckp on U2d's base build (today's graph, P1c2's rule), re-called under U2d's zrckp seed null.

# scr-u2g-dy-zrckp-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of the screen's p@swa minus the decision's base fit's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of U2d's own seeds 1 and 2 against the seed-0 fit on the base build) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | +0.0001 [-0.0008, +0.0011] | +0.0017, +0.0034 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | +0.0011 [-0.0011, +0.0034] | -0.0005, -0.0009 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | +0.0163 [+0.0101, +0.0225] | -0.0096, +0.0004 | 0.0136 | GAIN | GAIN |
| L-musique | hotpotqa | in-domain | -0.0542 [-0.0588, -0.0497] | -0.0005, +0.0011 | 0.0075 | LOSS | LOSS |
| L-musique | 2wiki | in-domain | -0.0174 [-0.0204, -0.0146] | -0.0020, +0.0017 | 0.0075 | LOSS | LOSS |
| L-musique | webqsp | zero-shot | -0.0038 [-0.0146, +0.0073] | +0.0035, +0.0124 | 0.0182 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | +0.0013 [+0.0005, +0.0022] | -0.0008, -0.0010 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | +0.0022 [-0.0000, +0.0045] | -0.0001, -0.0003 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | +0.0069 [+0.0007, +0.0131] | +0.0084, +0.0013 | 0.0120 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0483 [-0.0530, -0.0441] | -0.0036, -0.0046 | 0.0083 | LOSS | LOSS |
| L-hotpotqa | 2wiki | in-domain | -0.0085 [-0.0113, -0.0057] | +0.0036, +0.0021 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | webqsp | zero-shot | -0.0001 [-0.0105, +0.0105] | -0.0158, +0.0164 | 0.0322 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 1 of 12; with a LOSS: 4. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> MIXED; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: none.
Within 0.00015 of the floor (the call stands): none.
