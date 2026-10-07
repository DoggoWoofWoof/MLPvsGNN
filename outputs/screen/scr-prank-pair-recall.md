# scr-prank-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | -0.0310 [-0.0351, -0.0269] | +0.0008, -0.0023 | 0.0075 | LOSS | LOSS |
| L-musique | squad | in-domain | +0.0011 [-0.0017, +0.0037] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | +0.2016 [+0.1901, +0.2126] | +0.0503, -0.0079 | 0.0720 | GAIN | GAIN |
| L-musique | hotpotqa | in-domain | -0.0080 [-0.0108, -0.0054] | +0.0014, +0.0016 | 0.0075 | LOSS | LOSS |
| L-musique | 2wiki | in-domain | -0.0068 [-0.0091, -0.0047] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | +0.0269 [+0.0133, +0.0410] | -0.0070, -0.0018 | 0.0102 | GAIN | GAIN |
| L-hotpotqa | metaqa | in-domain | -0.0320 [-0.0362, -0.0278] | -0.0092, -0.0051 | 0.0149 | LOSS | LOSS |
| L-hotpotqa | squad | in-domain | -0.0035 [-0.0061, -0.0008] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | -0.0067 [-0.0133, -0.0000] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0255 [-0.0298, -0.0214] | -0.0055, -0.0130 | 0.0200 | LOSS | LOSS |
| L-hotpotqa | 2wiki | in-domain | -0.0004 [-0.0028, +0.0021] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | +0.0054 [-0.0051, +0.0167] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 2 of 12; with a LOSS: 4. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> MIXED; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: none.
Within 0.00015 of the floor (the call stands): none.
