A4he (docs/A4HE_SEMB_DROPOUT.md): zspkhe (zspk trained with A4he's dropout (0.1 per question on SEMB and with it rmatch's chain term, the eight fixed blocks kept), before its propagation) against zspk.

# scr-a4he-zspkhe-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of zspkhe's p@swa minus zspk's p@swa of the same split (zspk's card fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/A4HE_SEMB_DROPOUT.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zspk R@5 | zspkhe R@5 (A4he's SEMB dropout) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8009 | 0.8022 | +0.0013 [+0.0004, +0.0022] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9114 | 0.9093 | -0.0021 [-0.0040, -0.0003] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5012 | 0.3976 | -0.1036 [-0.1133, -0.0937] | +0.0503, -0.0079 | 0.0720 | LOSS | LOSS |
| L-musique | hotpotqa | in-domain | 0.9080 | 0.9072 | -0.0008 [-0.0029, +0.0012] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9383 | 0.9367 | -0.0016 [-0.0030, -0.0002] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3229 | 0.3190 | -0.0039 [-0.0130, +0.0051] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8040 | 0.8039 | -0.0001 [-0.0010, +0.0008] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9135 | 0.9132 | -0.0003 [-0.0023, +0.0014] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5754 | 0.5760 | +0.0006 [-0.0044, +0.0059] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8499 | 0.8598 | +0.0099 [+0.0068, +0.0130] | -0.0055, -0.0130 | 0.0200 | GAIN | **WITHIN** |
| L-hotpotqa | 2wiki | in-domain | 0.9333 | 0.9307 | -0.0025 [-0.0042, -0.0010] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2680 | 0.2992 | +0.0312 [+0.0211, +0.0415] | +0.0102, +0.0129 | 0.0233 | GAIN | GAIN |

Re-call: reads with a GAIN: 1 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa PROMISING -> PROMISING.
Changed calls: L-hotpotqa hotpotqa GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
