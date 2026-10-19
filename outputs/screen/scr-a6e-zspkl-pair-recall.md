A6e (docs/A6E_LINK_WORDS_EDGES.md): zspkl (zspk with each pool edge's link score as a weight in its propagation) against zspk.

# scr-a6e-zspkl-pair re-called under the seed null: **NO_GAIN** (filed: MIXED)

R@5 of zspkl's p@swa minus zspk's p@swa of the same split (zspk's card fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/A6E_LINK_WORDS_EDGES.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zspk R@5 | zspkl R@5 (A6e's link words) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8009 | 0.8020 | +0.0011 [+0.0002, +0.0020] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9114 | 0.9111 | -0.0003 [-0.0025, +0.0019] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5012 | 0.4918 | -0.0094 [-0.0149, -0.0042] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9080 | 0.9093 | +0.0013 [-0.0009, +0.0036] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9383 | 0.9331 | -0.0052 [-0.0067, -0.0035] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3229 | 0.3150 | -0.0079 [-0.0167, +0.0012] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8040 | 0.8051 | +0.0011 [+0.0001, +0.0022] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9135 | 0.9136 | +0.0001 [-0.0019, +0.0020] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5754 | 0.5825 | +0.0071 [+0.0014, +0.0123] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8499 | 0.8576 | +0.0077 [+0.0046, +0.0109] | -0.0055, -0.0130 | 0.0200 | GAIN | **WITHIN** |
| L-hotpotqa | 2wiki | in-domain | 0.9333 | 0.9403 | +0.0070 [+0.0054, +0.0086] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2680 | 0.2758 | +0.0078 [-0.0028, +0.0186] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa PROMISING -> NO_GAIN.
Changed calls: L-musique musique LOSS -> WITHIN; L-hotpotqa hotpotqa GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
