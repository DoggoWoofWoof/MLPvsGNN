A7 (docs/A7_S6_SCORER_INPUT.md): zrckps (zrckp plus the A7 head) against zrckp.

# scr-a7-zrckps-pair re-called under the seed null: **PROMISING** (filed: MIXED)

R@5 of zrckps's p@swa minus zrckp's p@swa of the same split (zrckp's card fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/A7_S6_SCORER_INPUT.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrckp R@5 | zrckps R@5 (A7's S6 scorer input) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8019 | 0.8043 | +0.0024 [+0.0013, +0.0034] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9116 | 0.9111 | -0.0006 [-0.0025, +0.0013] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5121 | 0.5133 | +0.0012 [-0.0044, +0.0069] | +0.0503, -0.0079 | 0.0720 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | 0.9065 | 0.9051 | -0.0014 [-0.0034, +0.0007] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 | 0.9362 | +0.0015 [-0.0000, +0.0031] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 | 0.3848 | +0.0221 [+0.0111, +0.0331] | -0.0070, -0.0018 | 0.0102 | GAIN | GAIN |
| L-hotpotqa | metaqa | in-domain | 0.8044 | 0.8042 | -0.0002 [-0.0011, +0.0007] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 | 0.9131 | +0.0003 [-0.0016, +0.0023] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 | 0.5737 | -0.0008 [-0.0056, +0.0041] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 | 0.8552 | +0.0038 [+0.0007, +0.0069] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 | 0.9300 | -0.0050 [-0.0066, -0.0032] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 | 0.3255 | -0.0219 [-0.0337, -0.0097] | +0.0102, +0.0129 | 0.0233 | LOSS | **WITHIN** |

Re-call: reads with a GAIN: 1 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique PROMISING -> PROMISING; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-hotpotqa webqsp LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
