A2 (docs/A2_RANKS_BESIDE_Z.md): zspka (zspk with A2's ranks and top-50 columns beside its z-scores, before its propagation) against zspk.

# scr-a2-zspka-pair re-called under the seed null: **NO_GAIN** (filed: MIXED)

R@5 of zspka's p@swa minus zspk's p@swa of the same split (zspk's card fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/A2_RANKS_BESIDE_Z.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zspk R@5 | zspka R@5 (A2's ranks beside) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8009 | 0.8007 | -0.0002 [-0.0013, +0.0008] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9114 | 0.9125 | +0.0011 [-0.0008, +0.0029] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5012 | 0.5260 | +0.0249 [+0.0191, +0.0312] | +0.0503, -0.0079 | 0.0720 | GAIN | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9080 | 0.9030 | -0.0050 [-0.0073, -0.0026] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9383 | 0.9285 | -0.0098 [-0.0113, -0.0082] | -0.0002, -0.0007 | 0.0075 | LOSS | LOSS |
| L-musique | webqsp | zero-shot | 0.3229 | 0.3116 | -0.0112 [-0.0231, -0.0004] | -0.0070, -0.0018 | 0.0102 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | 0.8040 | 0.7993 | -0.0046 [-0.0057, -0.0036] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9135 | 0.9126 | -0.0009 [-0.0028, +0.0009] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5754 | 0.5770 | +0.0016 [-0.0036, +0.0066] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8499 | 0.8498 | -0.0001 [-0.0031, +0.0030] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9333 | 0.9286 | -0.0047 [-0.0064, -0.0031] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2680 | 0.2692 | +0.0012 [-0.0088, +0.0113] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
