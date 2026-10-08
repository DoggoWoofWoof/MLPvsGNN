# Full run of gsurg re-graded under the seed null over every split: **ADOPT** (filed: NOT_ADOPTED)

R@5 of gsurg's p@swa minus step 1's p@swa of the same split, on the six s1eval carves. Every read is called with its floor from the seed null over all six splits (docs/SCREENS.md, section 2, amended 8 October). Bold: step 1's eleven primary reads. zs: read zero-shot. A changed call shows filed -> re-call.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0121 GAIN -> WITHIN (floor 0.0154)** | **-0.0018 WITHIN (floor 0.0075)** | **-0.0036 WITHIN (floor 0.0158)** | **+0.0019 WITHIN (floor 0.0075)** | **+0.0063 WITHIN (floor 0.0075)** | **+0.0008 WITHIN zs (floor 0.0295)** |
| L-metaqa | **-0.0025 WITHIN zs (floor 0.0168)** | +0.0008 WITHIN (floor 0.0075) | -0.0015 WITHIN (floor 0.0075) | -0.0007 WITHIN (floor 0.0075) | +0.0097 GAIN -> WITHIN (floor 0.0175) | -0.0020 WITHIN zs (floor 0.0075) |
| L-squad | +0.0026 WITHIN (floor 0.0075) | **+0.0036 WITHIN zs (floor 0.0075)** | -0.0094 LOSS -> WITHIN (floor 0.0239) | +0.0001 WITHIN (floor 0.0075) | +0.0023 WITHIN (floor 0.0117) | +0.0274 GAIN zs (floor 0.0260) |
| L-musique | +0.0012 WITHIN (floor 0.0075) | -0.0011 WITHIN (floor 0.0075) | **+0.0842 GAIN zs (floor 0.0720)** | +0.0003 WITHIN (floor 0.0075) | +0.0014 WITHIN (floor 0.0075) | -0.0007 WITHIN zs (floor 0.0102) |
| L-hotpotqa | +0.0016 WITHIN (floor 0.0149) | -0.0019 WITHIN (floor 0.0075) | +0.0049 WITHIN (floor 0.0075) | **-0.0147 LOSS -> WITHIN zs (floor 0.0200)** | +0.0091 GAIN (floor 0.0075) | +0.0124 GAIN -> WITHIN zs (floor 0.0233) |
| L-2wiki | +0.0103 GAIN (floor 0.0075) | -0.0035 WITHIN (floor 0.0099) | +0.0027 WITHIN (floor 0.0075) | +0.0055 WITHIN (floor 0.0089) | **-0.0072 WITHIN zs (floor 0.0112)** | +0.0123 GAIN -> WITHIN zs (floor 0.0473) |

Primary reads with a GAIN: 1. Reads with a LOSS: 0 of 36. Reads under the null's floors: 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs.
Changed calls: J5 metaqa GAIN -> WITHIN; L-metaqa 2wiki GAIN -> WITHIN; L-squad musique LOSS -> WITHIN; L-hotpotqa hotpotqa LOSS -> WITHIN; L-hotpotqa webqsp GAIN -> WITHIN; L-2wiki webqsp GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
