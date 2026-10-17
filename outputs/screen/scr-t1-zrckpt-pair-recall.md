T1 (docs/T1_GRANULARITY_TRANSFORMS.md): zrckpt (zrckp trained with metaqa's granularity transforms) against zrckp.

# scr-t1-zrckpt-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of zrckpt's p@swa minus zrckp's p@swa of the same split (zrckp's card fits, metaqa as filed), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/T1_GRANULARITY_TRANSFORMS.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrckp R@5 | zrckpt R@5 (T1's metaqa) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8019 | 0.7969 | -0.0050 [-0.0066, -0.0035] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9116 | 0.9109 | -0.0008 [-0.0025, +0.0011] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5121 | 0.4904 | -0.0217 [-0.0277, -0.0153] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9065 | 0.9053 | -0.0012 [-0.0033, +0.0008] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 | 0.9368 | +0.0021 [+0.0007, +0.0036] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 | 0.3475 | -0.0153 [-0.0280, -0.0029] | -0.0070, -0.0018 | 0.0102 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | 0.8044 | 0.7980 | -0.0065 [-0.0081, -0.0048] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 | 0.9119 | -0.0008 [-0.0027, +0.0011] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 | 0.5747 | +0.0002 [-0.0048, +0.0053] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 | 0.8582 | +0.0069 [+0.0040, +0.0100] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 | 0.9365 | +0.0015 [-0.0001, +0.0031] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 | 0.3132 | -0.0342 [-0.0476, -0.0203] | +0.0102, +0.0129 | 0.0233 | LOSS | LOSS |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
