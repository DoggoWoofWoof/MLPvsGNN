# scr-vshare-pair re-called under the seed null: **PROMISING** (filed: MIXED)

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | -0.0030 [-0.0056, -0.0002] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | -0.0004 [-0.0024, +0.0015] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | +0.0891 [+0.0793, +0.0990] | +0.0503, -0.0079 | 0.0720 | GAIN | GAIN |
| L-musique | hotpotqa | in-domain | -0.0017 [-0.0042, +0.0005] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | -0.0011 [-0.0030, +0.0010] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | +0.0045 [-0.0055, +0.0155] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | -0.0055 [-0.0087, -0.0024] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | +0.0003 [-0.0017, +0.0024] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | +0.0030 [-0.0027, +0.0091] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0140 [-0.0179, -0.0101] | -0.0055, -0.0130 | 0.0200 | LOSS | **WITHIN** |
| L-hotpotqa | 2wiki | in-domain | -0.0013 [-0.0035, +0.0010] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | -0.0057 [-0.0144, +0.0027] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 1 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique PROMISING -> PROMISING; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-hotpotqa hotpotqa LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
