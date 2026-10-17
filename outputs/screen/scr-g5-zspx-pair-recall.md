G5 (docs/G5_COMBINED_GNN.md): zspx (zspk with its step end to end, G3's residual message and G4's second step) against zspk.

# scr-g5-zspx-pair re-called under the seed null: **PROMISING** (filed: PROMISING)

R@5 of zspx's p@swa minus zspk's p@swa of the same split (zspk's card fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/G5_COMBINED_GNN.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zspk R@5 | zspx R@5 (zspk with its step end to end, G3's residual message and G4's second step) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8009 | 0.8030 | +0.0021 [+0.0011, +0.0031] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9114 | 0.9092 | -0.0022 [-0.0043, -0.0001] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5012 | 0.5073 | +0.0061 [+0.0006, +0.0119] | +0.0503, -0.0079 | 0.0720 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | 0.9080 | 0.9074 | -0.0007 [-0.0028, +0.0017] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9383 | 0.9351 | -0.0032 [-0.0045, -0.0019] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3229 | 0.3204 | -0.0025 [-0.0112, +0.0066] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8040 | 0.8039 | -0.0000 [-0.0009, +0.0009] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9135 | 0.9129 | -0.0006 [-0.0025, +0.0013] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5754 | 0.5901 | +0.0147 [+0.0095, +0.0198] | +0.0015, +0.0023 | 0.0075 | GAIN | GAIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8499 | 0.8547 | +0.0048 [+0.0020, +0.0076] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9333 | 0.9312 | -0.0020 [-0.0035, -0.0005] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2680 | 0.2914 | +0.0235 [+0.0133, +0.0327] | +0.0102, +0.0129 | 0.0233 | GAIN | GAIN |

Re-call: reads with a GAIN: 2 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa PROMISING -> PROMISING.
Changed calls: none.
Within 0.00015 of the floor (the call stands): none.
