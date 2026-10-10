# dsx-h2-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of zds's p@swa minus zrc's p@swa of the same split (zrc's screen fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, docs/DIAG_DATA_SCALE.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrc R@5 | subset R@5 | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7782 | 0.7739 | -0.0043 [-0.0059, -0.0028] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9105 | 0.9085 | -0.0019 [-0.0042, +0.0003] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5235 | 0.5165 | -0.0071 [-0.0120, -0.0019] | +0.0503, -0.0079 | 0.0720 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | 0.9041 | 0.8973 | -0.0068 [-0.0095, -0.0040] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.8733 | 0.8655 | -0.0078 [-0.0099, -0.0058] | -0.0002, -0.0007 | 0.0075 | LOSS | LOSS |
| L-musique | webqsp | zero-shot | 0.3384 | 0.3074 | -0.0310 [-0.0430, -0.0191] | -0.0070, -0.0018 | 0.0102 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | 0.7793 | 0.7729 | -0.0064 [-0.0079, -0.0049] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9112 | 0.9097 | -0.0015 [-0.0041, +0.0009] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5645 | 0.5681 | +0.0036 [-0.0029, +0.0098] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8451 | 0.8433 | -0.0018 [-0.0056, +0.0019] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.8731 | 0.8614 | -0.0118 [-0.0142, -0.0095] | -0.0020, +0.0011 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | webqsp | zero-shot | 0.2865 | 0.2484 | -0.0381 [-0.0505, -0.0256] | +0.0102, +0.0129 | 0.0233 | LOSS | LOSS |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 4. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: none.
Within 0.00015 of the floor (the call stands): none.
