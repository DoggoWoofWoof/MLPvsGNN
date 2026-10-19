A4f (docs/A4F_FAMILY_DROPOUT.md): zspkf (zspk trained with A4f's family dropout (0.2 per question on each block, the KT term and the chain term), before its propagation) against zspk.

# scr-a4f-zspkf-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of zspkf's p@swa minus zspk's p@swa of the same split (zspk's card fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/A4F_FAMILY_DROPOUT.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zspk R@5 | zspkf R@5 (A4f's block dropout) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8009 | 0.7970 | -0.0039 [-0.0055, -0.0025] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9114 | 0.9105 | -0.0009 [-0.0032, +0.0013] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5012 | 0.2748 | -0.2264 [-0.2372, -0.2158] | +0.0503, -0.0079 | 0.0720 | LOSS | LOSS |
| L-musique | hotpotqa | in-domain | 0.9080 | 0.9057 | -0.0023 [-0.0046, +0.0000] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9383 | 0.9328 | -0.0054 [-0.0070, -0.0039] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3229 | 0.3103 | -0.0125 [-0.0243, -0.0010] | -0.0070, -0.0018 | 0.0102 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | 0.8040 | 0.8023 | -0.0017 [-0.0029, -0.0003] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9135 | 0.9113 | -0.0022 [-0.0043, +0.0000] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5754 | 0.5775 | +0.0021 [-0.0034, +0.0076] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8499 | 0.8538 | +0.0039 [+0.0007, +0.0074] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9333 | 0.9363 | +0.0030 [+0.0013, +0.0047] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2680 | 0.2552 | -0.0128 [-0.0248, -0.0010] | +0.0102, +0.0129 | 0.0233 | LOSS | **WITHIN** |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-hotpotqa webqsp LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
