# Full run of zep re-graded under the seed null over every split: **NOT_ADOPTED** (filed: ADOPT)

R@5 of zep's p@swa minus the base arm's p@swa of the same split, on the six s1eval carves. Every read is called with its floor from the seed null over all six splits (docs/SCREENS.md, section 2, amended 8 October). Bold: step 1's eleven primary reads. zs: read zero-shot. A changed call shows filed -> re-call.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0020 WITHIN (floor 0.0154)** | **+0.0008 WITHIN (floor 0.0075)** | **+0.0043 WITHIN (floor 0.0158)** | **-0.0017 WITHIN (floor 0.0075)** | **+0.0028 WITHIN (floor 0.0075)** | **+0.0204 GAIN -> WITHIN zs (floor 0.0295)** |
| L-metaqa | **+0.0121 GAIN -> WITHIN zs (floor 0.0168)** | +0.0008 WITHIN (floor 0.0075) | -0.0027 WITHIN (floor 0.0075) | +0.0007 WITHIN (floor 0.0075) | +0.0018 WITHIN (floor 0.0175) | +0.0020 WITHIN zs (floor 0.0075) |
| L-squad | +0.0019 WITHIN (floor 0.0075) | **-0.0002 WITHIN zs (floor 0.0075)** | +0.0002 WITHIN (floor 0.0239) | +0.0028 WITHIN (floor 0.0075) | +0.0041 WITHIN (floor 0.0117) | +0.0145 GAIN -> WITHIN zs (floor 0.0260) |
| L-musique | +0.0021 WITHIN (floor 0.0075) | +0.0022 WITHIN (floor 0.0075) | **-0.0068 WITHIN zs (floor 0.0720)** | +0.0018 WITHIN (floor 0.0075) | +0.0017 WITHIN (floor 0.0075) | +0.0118 GAIN zs (floor 0.0102) |
| L-hotpotqa | +0.0026 WITHIN (floor 0.0149) | +0.0010 WITHIN (floor 0.0075) | -0.0066 WITHIN (floor 0.0075) | **-0.0021 WITHIN zs (floor 0.0200)** | +0.0004 WITHIN (floor 0.0075) | +0.0228 GAIN -> WITHIN zs (floor 0.0233) |
| L-2wiki | +0.0027 WITHIN (floor 0.0075) | -0.0001 WITHIN (floor 0.0099) | -0.0038 WITHIN (floor 0.0075) | +0.0052 WITHIN (floor 0.0089) | **+0.0046 WITHIN zs (floor 0.0112)** | +0.0122 GAIN -> WITHIN zs (floor 0.0473) |

Primary reads with a GAIN: 0. Reads with a LOSS: 0 of 36. Reads under the null's floors: 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs.
Changed calls: J5 webqsp GAIN -> WITHIN; L-metaqa metaqa GAIN -> WITHIN; L-squad webqsp GAIN -> WITHIN; L-hotpotqa webqsp GAIN -> WITHIN; L-2wiki webqsp GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
