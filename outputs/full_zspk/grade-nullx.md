# Full run of zspk re-graded under the seed null over every split: **ADOPT** (filed: NOT_ADOPTED)

R@5 of zspk's p@swa minus the base arm's p@swa of the same split, on the six s1eval carves. Every read is called with its floor from the seed null over all six splits (docs/SCREENS.md, section 2, amended 8 October). Bold: step 1's eleven primary reads. zs: read zero-shot. A changed call shows filed -> re-call.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0211 GAIN (floor 0.0154)** | **-0.0009 WITHIN (floor 0.0075)** | **+0.0110 GAIN -> WITHIN (floor 0.0158)** | **-0.0011 WITHIN (floor 0.0075)** | **+0.0586 GAIN (floor 0.0075)** | **+0.0331 GAIN zs (floor 0.0295)** |
| L-metaqa | **+0.0422 GAIN zs (floor 0.0168)** | -0.0029 WITHIN (floor 0.0075) | +0.0073 WITHIN (floor 0.0075) | +0.0010 WITHIN (floor 0.0075) | +0.0563 GAIN (floor 0.0175) | +0.0021 WITHIN zs (floor 0.0075) |
| L-squad | +0.0255 GAIN (floor 0.0075) | **+0.0015 WITHIN zs (floor 0.0075)** | +0.0075 WITHIN (floor 0.0239) | +0.0028 WITHIN (floor 0.0075) | +0.0533 GAIN (floor 0.0117) | +0.0263 GAIN zs (floor 0.0260) |
| L-musique | +0.0215 GAIN (floor 0.0075) | -0.0011 WITHIN (floor 0.0075) | **-0.0100 LOSS -> WITHIN zs (floor 0.0720)** | +0.0015 WITHIN (floor 0.0075) | +0.0659 GAIN (floor 0.0075) | -0.0017 WITHIN zs (floor 0.0102) |
| L-hotpotqa | +0.0233 GAIN (floor 0.0149) | +0.0026 WITHIN (floor 0.0075) | +0.0107 GAIN (floor 0.0075) | **+0.0064 WITHIN zs (floor 0.0200)** | +0.0589 GAIN (floor 0.0075) | +0.0418 GAIN zs (floor 0.0233) |
| L-2wiki | +0.0247 GAIN (floor 0.0075) | +0.0004 WITHIN (floor 0.0099) | +0.0070 WITHIN (floor 0.0075) | +0.0032 WITHIN (floor 0.0089) | **+0.0663 GAIN zs (floor 0.0112)** | +0.0351 GAIN -> WITHIN zs (floor 0.0473) |

Primary reads with a GAIN: 5. Reads with a LOSS: 0 of 36. Reads under the null's floors: 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs.
Changed calls: J5 musique GAIN -> WITHIN; L-musique musique LOSS -> WITHIN; L-2wiki webqsp GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
