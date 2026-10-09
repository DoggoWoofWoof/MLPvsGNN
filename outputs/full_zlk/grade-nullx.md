# Full run of zlk re-graded under the seed null over every split: **NOT_ADOPTED** (filed: NOT_ADOPTED)

R@5 of zlk's p@swa minus the base arm's p@swa of the same split, on the six s1eval carves. Every read is called with its floor from the seed null over all six splits (docs/SCREENS.md, section 2, amended 8 October). Bold: step 1's eleven primary reads. zs: read zero-shot. A changed call shows filed -> re-call.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **-0.0022 WITHIN (floor 0.0154)** | **+0.0000 WITHIN (floor 0.0075)** | **+0.0065 WITHIN (floor 0.0158)** | **+0.0038 WITHIN (floor 0.0075)** | **+0.0032 WITHIN (floor 0.0075)** | **+0.0008 WITHIN zs (floor 0.0295)** |
| L-metaqa | **-0.0014 WITHIN zs (floor 0.0168)** | +0.0015 WITHIN (floor 0.0075) | -0.0026 WITHIN (floor 0.0075) | +0.0031 WITHIN (floor 0.0075) | +0.0047 WITHIN (floor 0.0175) | -0.0042 WITHIN zs (floor 0.0075) |
| L-squad | -0.0022 WITHIN (floor 0.0075) | **-0.0013 WITHIN zs (floor 0.0075)** | +0.0092 GAIN -> WITHIN (floor 0.0239) | +0.0019 WITHIN (floor 0.0075) | +0.0073 WITHIN (floor 0.0117) | +0.0263 GAIN zs (floor 0.0260) |
| L-musique | -0.0044 WITHIN (floor 0.0075) | -0.0007 WITHIN (floor 0.0075) | **-0.0027 WITHIN zs (floor 0.0720)** | +0.0001 WITHIN (floor 0.0075) | -0.0041 WITHIN (floor 0.0075) | +0.0083 WITHIN zs (floor 0.0102) |
| L-hotpotqa | -0.0014 WITHIN (floor 0.0149) | +0.0019 WITHIN (floor 0.0075) | +0.0084 GAIN (floor 0.0075) | **+0.0011 WITHIN zs (floor 0.0200)** | +0.0047 WITHIN (floor 0.0075) | +0.0055 WITHIN zs (floor 0.0233) |
| L-2wiki | -0.0021 WITHIN (floor 0.0075) | +0.0006 WITHIN (floor 0.0099) | +0.0008 WITHIN (floor 0.0075) | +0.0000 WITHIN (floor 0.0089) | **-0.0179 LOSS zs (floor 0.0112)** | +0.0095 WITHIN zs (floor 0.0473) |

Primary reads with a GAIN: 0. Reads with a LOSS: 1 of 36. Reads under the null's floors: 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs.
Changed calls: L-squad musique GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
