# Full run of zspx re-graded under the seed null over every split: **NOT_ADOPTED** (filed: NOT_ADOPTED)

R@5 of zspx's p@swa minus the base arm's p@swa of the same split, on the six s1eval carves. Every read is called with its floor from the seed null over all six splits (docs/SCREENS.md, section 2, amended 8 October). Bold: step 1's eleven primary reads. zs: read zero-shot. A changed call shows filed -> re-call.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0009 WITHIN (floor 0.0154)** | **-0.0003 WITHIN (floor 0.0075)** | **+0.0060 WITHIN (floor 0.0158)** | **-0.0030 WITHIN (floor 0.0075)** | **-0.0011 WITHIN (floor 0.0075)** | **-0.0099 LOSS -> WITHIN zs (floor 0.0295)** |
| L-metaqa | **+0.0130 GAIN -> WITHIN zs (floor 0.0168)** | +0.0007 WITHIN (floor 0.0075) | -0.0007 WITHIN (floor 0.0075) | -0.0014 WITHIN (floor 0.0075) | +0.0005 WITHIN (floor 0.0175) | +0.0003 WITHIN zs (floor 0.0075) |
| L-squad | +0.0030 WITHIN (floor 0.0075) | **-0.0053 WITHIN zs (floor 0.0075)** | +0.0016 WITHIN (floor 0.0239) | +0.0036 WITHIN (floor 0.0075) | +0.0046 WITHIN (floor 0.0117) | -0.0187 LOSS -> WITHIN zs (floor 0.0260) |
| L-musique | +0.0018 WITHIN (floor 0.0075) | -0.0010 WITHIN (floor 0.0075) | **-0.0086 LOSS -> WITHIN zs (floor 0.0720)** | +0.0018 WITHIN (floor 0.0075) | +0.0083 GAIN (floor 0.0075) | -0.0125 LOSS zs (floor 0.0102) |
| L-hotpotqa | +0.0003 WITHIN (floor 0.0149) | -0.0002 WITHIN (floor 0.0075) | +0.0054 WITHIN (floor 0.0075) | **+0.0086 GAIN -> WITHIN zs (floor 0.0200)** | +0.0022 WITHIN (floor 0.0075) | +0.0174 GAIN -> WITHIN zs (floor 0.0233) |
| L-2wiki | +0.0033 WITHIN (floor 0.0075) | +0.0005 WITHIN (floor 0.0099) | +0.0021 WITHIN (floor 0.0075) | +0.0016 WITHIN (floor 0.0089) | **+0.0023 WITHIN zs (floor 0.0112)** | +0.0013 WITHIN zs (floor 0.0473) |

Primary reads with a GAIN: 0. Reads with a LOSS: 1 of 36. Reads under the null's floors: 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs.
Changed calls: J5 webqsp LOSS -> WITHIN; L-metaqa metaqa GAIN -> WITHIN; L-squad webqsp LOSS -> WITHIN; L-musique musique LOSS -> WITHIN; L-hotpotqa hotpotqa GAIN -> WITHIN; L-hotpotqa webqsp GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
