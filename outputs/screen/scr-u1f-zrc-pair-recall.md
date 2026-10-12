U1f (docs/U1F_RETRAIN_ON_GU1024.md; U1c's screen on GU_1024): zrc on GU_1024 and its step 4h pools against zrc on today's.

# scr-u1f-zrc-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of zrc's p@swa minus zrc's p@swa of the same split (zrc's card fits on today's graph and pools), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/U1C_RETRAIN_ON_U.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrc R@5 (today's) | zrc R@5 (U1f) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7782 | 0.7857 | +0.0074 [+0.0054, +0.0095] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9105 | 0.9117 | +0.0013 [-0.0008, +0.0034] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5235 | 0.4977 | -0.0259 [-0.0342, -0.0178] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9041 | 0.8454 | -0.0586 [-0.0631, -0.0538] | +0.0014, +0.0016 | 0.0075 | LOSS | LOSS |
| L-musique | 2wiki | in-domain | 0.8733 | 0.8635 | -0.0098 [-0.0129, -0.0065] | -0.0002, -0.0007 | 0.0075 | LOSS | LOSS |
| L-musique | webqsp | zero-shot | 0.3384 | 0.3421 | +0.0037 [-0.0106, +0.0174] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.7793 | 0.7852 | +0.0059 [+0.0039, +0.0079] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9112 | 0.9135 | +0.0023 [+0.0000, +0.0045] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5645 | 0.5547 | -0.0098 [-0.0167, -0.0030] | +0.0015, +0.0023 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | hotpotqa | zero-shot | 0.8451 | 0.7999 | -0.0452 [-0.0499, -0.0406] | -0.0055, -0.0130 | 0.0200 | LOSS | LOSS |
| L-hotpotqa | 2wiki | in-domain | 0.8731 | 0.8623 | -0.0108 [-0.0141, -0.0074] | -0.0020, +0.0011 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | webqsp | zero-shot | 0.2865 | 0.3007 | +0.0142 [-0.0001, +0.0286] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 5. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
