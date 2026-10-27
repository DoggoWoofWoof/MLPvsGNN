A6ex (docs/A6EX_LINK_WORDS_ADOPTED.md): zrckprn (zrckpr plus A6e's head over the eight link-word columns) against zrckpr.

# scr-a6ex-zrckprn-pair re-called under the seed null: **NO_GAIN** (filed: MIXED)

R@5 of zrckprn's p@swa minus zrckpr's p@swa of the same split (zrckpr's card fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/A6EX_LINK_WORDS_ADOPTED.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrckpr R@5 | zrckprn R@5 (A6e's link words) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8012 | 0.8016 | +0.0003 [-0.0005, +0.0012] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9127 | 0.9125 | -0.0003 [-0.0022, +0.0017] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5197 | 0.5099 | -0.0098 [-0.0151, -0.0048] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9083 | 0.9097 | +0.0014 [-0.0008, +0.0036] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9361 | 0.9308 | -0.0053 [-0.0068, -0.0036] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3762 | 0.3644 | -0.0118 [-0.0202, -0.0033] | -0.0070, -0.0018 | 0.0102 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | 0.8040 | 0.8032 | -0.0009 [-0.0016, -0.0001] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9114 | 0.9129 | +0.0015 [-0.0005, +0.0036] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5779 | 0.5801 | +0.0022 [-0.0027, +0.0074] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8558 | 0.8619 | +0.0061 [+0.0032, +0.0089] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9331 | 0.9358 | +0.0026 [+0.0011, +0.0041] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3411 | 0.3526 | +0.0115 [+0.0025, +0.0215] | +0.0102, +0.0129 | 0.0233 | GAIN | **WITHIN** |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa PROMISING -> NO_GAIN.
Changed calls: L-musique musique LOSS -> WITHIN; L-hotpotqa webqsp GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
