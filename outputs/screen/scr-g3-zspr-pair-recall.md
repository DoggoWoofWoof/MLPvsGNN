G3 (docs/G3_BRIDGE_RESIDUAL.md): zspr (zspk plus the bridge-residual head) against zspk.

# scr-g3-zspr-pair re-called under the seed null: **PROMISING** (filed: PROMISING)

R@5 of zspr's p@swa minus zspk's p@swa of the same split (zspk's card fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/G3_BRIDGE_RESIDUAL.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zspk R@5 | zspr R@5 (G3's bridge residual) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8009 | 0.8021 | +0.0012 [+0.0003, +0.0020] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9114 | 0.9110 | -0.0004 [-0.0024, +0.0015] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5012 | 0.5125 | +0.0114 [+0.0056, +0.0174] | +0.0503, -0.0079 | 0.0720 | GAIN | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9080 | 0.9083 | +0.0003 [-0.0017, +0.0024] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9383 | 0.9365 | -0.0018 [-0.0031, -0.0005] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3229 | 0.3191 | -0.0038 [-0.0128, +0.0046] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8040 | 0.8043 | +0.0003 [-0.0006, +0.0012] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9135 | 0.9132 | -0.0003 [-0.0023, +0.0015] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5754 | 0.5850 | +0.0096 [+0.0043, +0.0148] | +0.0015, +0.0023 | 0.0075 | GAIN | GAIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8499 | 0.8470 | -0.0029 [-0.0057, +0.0000] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9333 | 0.9358 | +0.0025 [+0.0010, +0.0039] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2680 | 0.2812 | +0.0132 [+0.0025, +0.0235] | +0.0102, +0.0129 | 0.0233 | GAIN | **WITHIN** |

Re-call: reads with a GAIN: 1 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique PROMISING -> NO_GAIN; L-hotpotqa PROMISING -> PROMISING.
Changed calls: L-musique musique GAIN -> WITHIN; L-hotpotqa webqsp GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
