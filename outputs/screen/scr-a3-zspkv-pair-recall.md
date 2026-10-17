A3 (docs/A3_VSHARE_LEAN.md): zspkv (zspk trained with A3's share augmentation of zret's reference, before its propagation) against zspk.

# scr-a3-zspkv-pair re-called under the seed null: **NO_GAIN** (filed: MIXED)

R@5 of zspkv's p@swa minus zspk's p@swa of the same split (zspk's card fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/A3_VSHARE_LEAN.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zspk R@5 | zspkv R@5 (A3's share augmentation) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8009 | 0.8016 | +0.0007 [-0.0002, +0.0016] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9114 | 0.9101 | -0.0013 [-0.0031, +0.0007] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5012 | 0.5198 | +0.0186 [+0.0131, +0.0243] | +0.0503, -0.0079 | 0.0720 | GAIN | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9080 | 0.9077 | -0.0003 [-0.0024, +0.0018] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9383 | 0.9345 | -0.0038 [-0.0052, -0.0024] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3229 | 0.3125 | -0.0103 [-0.0198, -0.0012] | -0.0070, -0.0018 | 0.0102 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | 0.8040 | 0.8022 | -0.0018 [-0.0028, -0.0008] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9135 | 0.9121 | -0.0014 [-0.0033, +0.0005] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5754 | 0.5826 | +0.0072 [+0.0021, +0.0125] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8499 | 0.8490 | -0.0009 [-0.0039, +0.0020] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9333 | 0.9319 | -0.0014 [-0.0031, +0.0003] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2680 | 0.2594 | -0.0086 [-0.0191, +0.0015] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): L-musique webqsp (-0.0103 against 0.0102).
