# scr-zof-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of zof's p@swa minus zrc's p@swa of the same split (zrc's screen fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2 and the thirty-second round). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrc R@5 | zof R@5 | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7782 | 0.7826 | +0.0043 [+0.0028, +0.0058] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9105 | 0.8935 | -0.0169 [-0.0203, -0.0136] | +0.0001, +0.0013 | 0.0075 | LOSS | LOSS |
| L-musique | musique | zero-shot | 0.5235 | 0.4858 | -0.0377 [-0.0459, -0.0303] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9041 | 0.8956 | -0.0084 [-0.0115, -0.0052] | +0.0014, +0.0016 | 0.0075 | LOSS | LOSS |
| L-musique | 2wiki | in-domain | 0.8733 | 0.8714 | -0.0019 [-0.0042, +0.0005] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3384 | 0.3351 | -0.0033 [-0.0156, +0.0088] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.7793 | 0.7839 | +0.0046 [+0.0032, +0.0061] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9112 | 0.8945 | -0.0168 [-0.0201, -0.0131] | +0.0002, +0.0007 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | musique | in-domain | 0.5645 | 0.4771 | -0.0873 [-0.0964, -0.0780] | +0.0015, +0.0023 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | hotpotqa | zero-shot | 0.8451 | 0.7957 | -0.0494 [-0.0543, -0.0444] | -0.0055, -0.0130 | 0.0200 | LOSS | LOSS |
| L-hotpotqa | 2wiki | in-domain | 0.8731 | 0.8614 | -0.0117 [-0.0143, -0.0089] | -0.0020, +0.0011 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | webqsp | zero-shot | 0.2865 | 0.2618 | -0.0247 [-0.0377, -0.0115] | +0.0102, +0.0129 | 0.0233 | LOSS | LOSS |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 7. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
