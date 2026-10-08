# Full run of rel re-graded under the seed null over every split: **NOT_ADOPTED** (filed: NOT_ADOPTED)

R@5 of rel's p@swa minus step 1's p@swa of the same split, on the six s1eval carves. Every read is called with its floor from the seed null over all six splits (docs/SCREENS.md, section 2, amended 8 October). Bold: step 1's eleven primary reads. zs: read zero-shot. A changed call shows filed -> re-call.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0759 GAIN (floor 0.0154)** | **-0.0013 WITHIN (floor 0.0075)** | **-0.0013 WITHIN (floor 0.0158)** | **+0.0019 WITHIN (floor 0.0075)** | **+0.0001 WITHIN (floor 0.0075)** | **+0.1013 GAIN zs (floor 0.0295)** |
| L-metaqa | **+0.0000 WITHIN zs (floor 0.0168)** | +0.0000 WITHIN (floor 0.0075) | +0.0000 WITHIN (floor 0.0075) | +0.0000 WITHIN (floor 0.0075) | +0.0000 WITHIN (floor 0.0175) | +0.0000 WITHIN zs (floor 0.0075) |
| L-squad | +0.0721 GAIN (floor 0.0075) | **+0.0013 WITHIN zs (floor 0.0075)** | -0.0093 LOSS -> WITHIN (floor 0.0239) | -0.0010 WITHIN (floor 0.0075) | +0.0022 WITHIN (floor 0.0117) | +0.1127 GAIN zs (floor 0.0260) |
| L-musique | +0.0716 GAIN (floor 0.0075) | -0.0008 WITHIN (floor 0.0075) | **+0.0868 GAIN zs (floor 0.0720)** | +0.0005 WITHIN (floor 0.0075) | -0.0012 WITHIN (floor 0.0075) | +0.0635 GAIN zs (floor 0.0102) |
| L-hotpotqa | +0.0677 GAIN (floor 0.0149) | -0.0012 WITHIN (floor 0.0075) | +0.0053 WITHIN (floor 0.0075) | **-0.0132 LOSS -> WITHIN zs (floor 0.0200)** | +0.0015 WITHIN (floor 0.0075) | +0.0702 GAIN zs (floor 0.0233) |
| L-2wiki | +0.0795 GAIN (floor 0.0075) | -0.0025 WITHIN (floor 0.0099) | -0.0033 WITHIN (floor 0.0075) | +0.0059 WITHIN (floor 0.0089) | **-0.0136 LOSS zs (floor 0.0112)** | +0.1156 GAIN zs (floor 0.0473) |

Primary reads with a GAIN: 3. Reads with a LOSS: 1 of 36. Reads under the null's floors: 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs.
Changed calls: L-squad musique LOSS -> WITHIN; L-hotpotqa hotpotqa LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
