# zkind's fits against zrc's (the next base), re-graded under the seed null over every split: **NOT_ADOPTED** (filed: NOT_ADOPTED)

R@5 of zkind's p@swa minus the base arm's p@swa of the same split, on the six s1eval carves. Every read is called with its floor from the seed null over all six splits (docs/SCREENS.md, section 2, amended 8 October). Bold: step 1's eleven primary reads. zs: read zero-shot. A changed call shows filed -> re-call.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0001 WITHIN (floor 0.0154)** | **+0.0008 WITHIN (floor 0.0075)** | **+0.0031 WITHIN (floor 0.0158)** | **-0.0063 WITHIN (floor 0.0075)** | **-0.0037 WITHIN (floor 0.0075)** | **-0.0141 WITHIN zs (floor 0.0295)** |
| L-metaqa | **+0.0000 WITHIN zs (floor 0.0168)** | +0.0000 WITHIN (floor 0.0075) | +0.0000 WITHIN (floor 0.0075) | +0.0000 WITHIN (floor 0.0075) | +0.0000 WITHIN (floor 0.0175) | +0.0000 WITHIN zs (floor 0.0075) |
| L-squad | -0.0013 WITHIN (floor 0.0075) | **-0.0075 WITHIN zs (floor 0.0075)** | -0.0011 WITHIN (floor 0.0239) | +0.0018 WITHIN (floor 0.0075) | +0.0019 WITHIN (floor 0.0117) | +0.0054 WITHIN zs (floor 0.0260) |
| L-musique | +0.0004 WITHIN (floor 0.0075) | +0.0005 WITHIN (floor 0.0075) | **+0.0096 GAIN -> WITHIN zs (floor 0.0720)** | -0.0013 WITHIN (floor 0.0075) | -0.0046 WITHIN (floor 0.0075) | -0.0089 WITHIN zs (floor 0.0102) |
| L-hotpotqa | -0.0014 WITHIN (floor 0.0149) | +0.0028 WITHIN (floor 0.0075) | +0.0033 WITHIN (floor 0.0075) | **-0.0107 LOSS -> WITHIN zs (floor 0.0200)** | -0.0044 WITHIN (floor 0.0075) | +0.0142 WITHIN zs (floor 0.0233) |
| L-2wiki | +0.0002 WITHIN (floor 0.0075) | -0.0004 WITHIN (floor 0.0099) | +0.0018 WITHIN (floor 0.0075) | +0.0014 WITHIN (floor 0.0089) | **+0.0026 WITHIN zs (floor 0.0112)** | -0.0070 WITHIN zs (floor 0.0473) |

Primary reads with a GAIN: 0. Reads with a LOSS: 0 of 36. Reads under the null's floors: 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs.
Changed calls: L-musique musique GAIN -> WITHIN; L-hotpotqa hotpotqa LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
