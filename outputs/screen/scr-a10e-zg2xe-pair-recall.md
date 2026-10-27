A10e (docs/A10E_G2_END_TO_END.md): zg2xe (zspx with G2's hidden-state layers over its score, trained end to end) against zspx.

# scr-a10e-zg2xe-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of zg2xe's p@swa minus zspx's p@swa of the same split (zspx's card fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/A10E_G2_END_TO_END.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zspx R@5 | zg2xe R@5 (zspx with G2's hidden-state layers over its score, trained end to end) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8030 | 0.8031 | +0.0001 [-0.0007, +0.0009] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9092 | 0.9116 | +0.0024 [+0.0005, +0.0042] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5073 | 0.4903 | -0.0169 [-0.0225, -0.0114] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9074 | 0.9085 | +0.0011 [-0.0010, +0.0031] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9351 | 0.9297 | -0.0054 [-0.0068, -0.0039] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3204 | 0.3167 | -0.0037 [-0.0116, +0.0040] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8039 | 0.8027 | -0.0012 [-0.0020, -0.0005] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9129 | 0.9138 | +0.0008 [-0.0010, +0.0028] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5901 | 0.5860 | -0.0041 [-0.0088, +0.0008] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8547 | 0.8537 | -0.0010 [-0.0035, +0.0016] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9312 | 0.9347 | +0.0034 [+0.0020, +0.0049] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2914 | 0.2834 | -0.0080 [-0.0158, +0.0002] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
