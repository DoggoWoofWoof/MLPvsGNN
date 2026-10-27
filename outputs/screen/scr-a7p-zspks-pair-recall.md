A7p (docs/A7P_P1GB_POOLS.md): zspks (zspk plus the A7 head before its propagation) against zspk, both on P1g-b's pools.

# scr-a7p-zspks-pair re-called under the seed null: **PROMISING** (filed: PROMISING)

R@5 of zspks's p@swa minus zspk's p@swa of the same split (zspk's card fits on P1g-b's pools), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/A7P_P1GB_POOLS.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zspk R@5 (P1g-b's) | zspks R@5 (A7's scorer input, P1g-b's) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8004 | 0.8014 | +0.0010 [-0.0001, +0.0022] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9129 | 0.9142 | +0.0013 [-0.0004, +0.0029] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5169 | 0.5180 | +0.0012 [-0.0042, +0.0066] | +0.0503, -0.0079 | 0.0720 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | 0.9086 | 0.9103 | +0.0017 [-0.0006, +0.0040] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9344 | 0.9388 | +0.0044 [+0.0030, +0.0059] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3354 | 0.3389 | +0.0036 [-0.0060, +0.0132] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8011 | 0.8016 | +0.0005 [-0.0006, +0.0016] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 | 0.9142 | +0.0015 [-0.0003, +0.0033] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5662 | 0.5746 | +0.0084 [+0.0031, +0.0135] | +0.0015, +0.0023 | 0.0075 | GAIN | GAIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8483 | 0.8533 | +0.0049 [+0.0022, +0.0076] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9326 | 0.9356 | +0.0030 [+0.0015, +0.0046] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2808 | 0.2851 | +0.0042 [-0.0056, +0.0141] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 1 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa PROMISING -> PROMISING.
Changed calls: none.
Within 0.00015 of the floor (the call stands): none.
