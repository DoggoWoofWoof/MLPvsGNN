# Full run of zrg re-graded under the seed null over every split: **NOT_ADOPTED** (filed: NOT_ADOPTED)

R@5 of zrg's p@swa minus the base arm's p@swa of the same split, on the six s1eval carves. Every read is called with its floor from the seed null over all six splits (docs/SCREENS.md, section 2, amended 8 October). Bold: step 1's eleven primary reads. zs: read zero-shot. A changed call shows filed -> re-call.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0017 WITHIN (floor 0.0154)** | **-0.0002 WITHIN (floor 0.0075)** | **+0.0058 WITHIN (floor 0.0158)** | **+0.0051 WITHIN (floor 0.0075)** | **+0.0026 WITHIN (floor 0.0075)** | **+0.0138 GAIN -> WITHIN zs (floor 0.0295)** |
| L-metaqa | **+0.0027 WITHIN zs (floor 0.0168)** | +0.0001 WITHIN (floor 0.0075) | -0.0045 WITHIN (floor 0.0075) | +0.0018 WITHIN (floor 0.0075) | +0.0012 WITHIN (floor 0.0175) | +0.0058 WITHIN zs (floor 0.0075) |
| L-squad | +0.0008 WITHIN (floor 0.0075) | **-0.0032 WITHIN zs (floor 0.0075)** | -0.0001 WITHIN (floor 0.0239) | +0.0028 WITHIN (floor 0.0075) | +0.0034 WITHIN (floor 0.0117) | +0.0273 GAIN zs (floor 0.0260) |
| L-musique | +0.0012 WITHIN (floor 0.0075) | +0.0016 WITHIN (floor 0.0075) | **-0.0040 WITHIN zs (floor 0.0720)** | +0.0022 WITHIN (floor 0.0075) | +0.0027 WITHIN (floor 0.0075) | +0.0018 WITHIN zs (floor 0.0102) |
| L-hotpotqa | +0.0003 WITHIN (floor 0.0149) | +0.0008 WITHIN (floor 0.0075) | +0.0087 GAIN (floor 0.0075) | **-0.0094 LOSS -> WITHIN zs (floor 0.0200)** | +0.0053 WITHIN (floor 0.0075) | +0.0081 WITHIN zs (floor 0.0233) |
| L-2wiki | +0.0014 WITHIN (floor 0.0075) | +0.0008 WITHIN (floor 0.0099) | -0.0060 WITHIN (floor 0.0075) | +0.0021 WITHIN (floor 0.0089) | **-0.0068 WITHIN zs (floor 0.0112)** | +0.0041 WITHIN zs (floor 0.0473) |

Primary reads with a GAIN: 0. Reads with a LOSS: 0 of 36. Reads under the null's floors: 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs.
Changed calls: J5 webqsp GAIN -> WITHIN; L-hotpotqa hotpotqa LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
