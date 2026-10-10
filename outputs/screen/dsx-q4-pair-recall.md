# dsx-q4-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of zds's p@swa minus zrc's p@swa of the same split (zrc's screen fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, docs/DIAG_DATA_SCALE.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrc R@5 | subset R@5 | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7782 | 0.7523 | -0.0259 [-0.0286, -0.0235] | +0.0008, -0.0023 | 0.0075 | LOSS | LOSS |
| L-musique | squad | in-domain | 0.9105 | 0.9020 | -0.0084 [-0.0115, -0.0055] | +0.0001, +0.0013 | 0.0075 | LOSS | LOSS |
| L-musique | musique | zero-shot | 0.5235 | 0.5056 | -0.0180 [-0.0246, -0.0117] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9041 | 0.8895 | -0.0146 [-0.0177, -0.0113] | +0.0014, +0.0016 | 0.0075 | LOSS | LOSS |
| L-musique | 2wiki | in-domain | 0.8733 | 0.8558 | -0.0175 [-0.0199, -0.0151] | -0.0002, -0.0007 | 0.0075 | LOSS | LOSS |
| L-musique | webqsp | zero-shot | 0.3384 | 0.2571 | -0.0813 [-0.0978, -0.0654] | -0.0070, -0.0018 | 0.0102 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | 0.7793 | 0.7495 | -0.0298 [-0.0327, -0.0272] | -0.0092, -0.0051 | 0.0149 | LOSS | LOSS |
| L-hotpotqa | squad | in-domain | 0.9112 | 0.9023 | -0.0089 [-0.0118, -0.0060] | +0.0002, +0.0007 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | musique | in-domain | 0.5645 | 0.5552 | -0.0093 [-0.0161, -0.0021] | +0.0015, +0.0023 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | hotpotqa | zero-shot | 0.8451 | 0.8367 | -0.0084 [-0.0126, -0.0046] | -0.0055, -0.0130 | 0.0200 | LOSS | **WITHIN** |
| L-hotpotqa | 2wiki | in-domain | 0.8731 | 0.8489 | -0.0242 [-0.0270, -0.0216] | -0.0020, +0.0011 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | webqsp | zero-shot | 0.2865 | 0.2323 | -0.0542 [-0.0680, -0.0388] | +0.0102, +0.0129 | 0.0233 | LOSS | LOSS |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 10. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique LOSS -> WITHIN; L-hotpotqa hotpotqa LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
