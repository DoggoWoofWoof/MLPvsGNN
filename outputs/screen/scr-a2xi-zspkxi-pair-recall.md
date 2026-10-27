A2xi (docs/A2XI_PRANK_REPLACE.md): zspkxi (zspk with each fixed block's z-score replaced by prank's hi and lo, before its propagation) against zspk.

# scr-a2xi-zspkxi-pair re-called under the seed null: **NO_GAIN** (filed: MIXED)

R@5 of zspkxi's p@swa minus zspk's p@swa of the same split (zspk's card fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/A2XI_PRANK_REPLACE.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zspk R@5 | zspkxi R@5 (A2xi's ranks in the z-scores' place) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8009 | 0.8036 | +0.0027 [+0.0016, +0.0038] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9114 | 0.9109 | -0.0005 [-0.0030, +0.0019] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5012 | 0.5122 | +0.0110 [+0.0043, +0.0180] | +0.0503, -0.0079 | 0.0720 | GAIN | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9080 | 0.9026 | -0.0054 [-0.0080, -0.0027] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9383 | 0.9339 | -0.0044 [-0.0061, -0.0027] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3229 | 0.2993 | -0.0236 [-0.0356, -0.0115] | -0.0070, -0.0018 | 0.0102 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | 0.8040 | 0.7937 | -0.0103 [-0.0117, -0.0089] | -0.0092, -0.0051 | 0.0149 | LOSS | **WITHIN** |
| L-hotpotqa | squad | in-domain | 0.9135 | 0.9106 | -0.0029 [-0.0053, -0.0006] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5754 | 0.5703 | -0.0051 [-0.0109, +0.0003] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8499 | 0.8367 | -0.0132 [-0.0167, -0.0097] | -0.0055, -0.0130 | 0.0200 | LOSS | **WITHIN** |
| L-hotpotqa | 2wiki | in-domain | 0.9333 | 0.9348 | +0.0015 [-0.0003, +0.0033] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2680 | 0.2747 | +0.0068 [-0.0078, +0.0204] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique GAIN -> WITHIN; L-hotpotqa metaqa LOSS -> WITHIN; L-hotpotqa hotpotqa LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
