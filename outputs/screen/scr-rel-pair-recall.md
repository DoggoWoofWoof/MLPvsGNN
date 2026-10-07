# scr-rel-pair re-called under the seed null: **PROMISING** (filed: MIXED)

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | +0.0716 [+0.0672, +0.0761] | +0.0008, -0.0023 | 0.0075 | GAIN | GAIN |
| L-musique | squad | in-domain | -0.0008 [-0.0029, +0.0013] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | +0.0868 [+0.0773, +0.0965] | +0.0503, -0.0079 | 0.0720 | GAIN | GAIN |
| L-musique | hotpotqa | in-domain | +0.0005 [-0.0019, +0.0028] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | -0.0012 [-0.0029, +0.0006] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | +0.0635 [+0.0477, +0.0794] | -0.0070, -0.0018 | 0.0102 | GAIN | GAIN |
| L-hotpotqa | metaqa | in-domain | +0.0677 [+0.0633, +0.0719] | -0.0092, -0.0051 | 0.0149 | GAIN | GAIN |
| L-hotpotqa | squad | in-domain | -0.0012 [-0.0032, +0.0009] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | +0.0053 [-0.0006, +0.0112] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0132 [-0.0171, -0.0095] | -0.0055, -0.0130 | 0.0200 | LOSS | **WITHIN** |
| L-hotpotqa | 2wiki | in-domain | +0.0015 [-0.0006, +0.0036] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | +0.0702 [+0.0514, +0.0893] | +0.0102, +0.0129 | 0.0233 | GAIN | GAIN |

Re-call: reads with a GAIN: 5 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique PROMISING -> PROMISING; L-hotpotqa MIXED -> PROMISING.
Changed calls: L-hotpotqa hotpotqa LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
