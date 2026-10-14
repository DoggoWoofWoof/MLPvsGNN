# Full run of zrck re-graded under the seed null over every split: **ADOPT** (filed: NOT_ADOPTED)

R@5 of zrck's p@swa minus the base arm's p@swa of the same split, on the six s1eval carves. Every read is called with its floor from the seed null over all six splits (docs/SCREENS.md, section 2, amended 8 October). Bold: step 1's eleven primary reads. zs: read zero-shot. A changed call shows filed -> re-call.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0196 GAIN (floor 0.0154)** | **+0.0001 WITHIN (floor 0.0075)** | **+0.0019 WITHIN (floor 0.0158)** | **-0.0012 WITHIN (floor 0.0075)** | **+0.0573 GAIN (floor 0.0075)** | **+0.0529 GAIN zs (floor 0.0295)** |
| L-metaqa | **+0.0619 GAIN zs (floor 0.0168)** | +0.0003 WITHIN (floor 0.0075) | +0.0042 WITHIN (floor 0.0075) | +0.0063 WITHIN (floor 0.0075) | +0.0662 GAIN (floor 0.0175) | +0.0212 GAIN zs (floor 0.0075) |
| L-squad | +0.0216 GAIN (floor 0.0075) | **-0.0028 WITHIN zs (floor 0.0075)** | +0.0061 WITHIN (floor 0.0239) | +0.0034 WITHIN (floor 0.0075) | +0.0623 GAIN (floor 0.0117) | +0.0721 GAIN zs (floor 0.0260) |
| L-musique | +0.0225 GAIN (floor 0.0075) | +0.0013 WITHIN (floor 0.0075) | **-0.0176 LOSS -> WITHIN zs (floor 0.0720)** | +0.0022 WITHIN (floor 0.0075) | +0.0632 GAIN (floor 0.0075) | +0.0372 GAIN zs (floor 0.0102) |
| L-hotpotqa | +0.0259 GAIN (floor 0.0149) | +0.0009 WITHIN (floor 0.0075) | +0.0050 WITHIN (floor 0.0075) | **+0.0113 GAIN -> WITHIN zs (floor 0.0200)** | +0.0397 GAIN (floor 0.0075) | +0.0820 GAIN zs (floor 0.0233) |
| L-2wiki | +0.0215 GAIN (floor 0.0075) | -0.0019 WITHIN (floor 0.0099) | +0.0039 WITHIN (floor 0.0075) | +0.0045 WITHIN (floor 0.0089) | **+0.0639 GAIN zs (floor 0.0112)** | +0.0548 GAIN zs (floor 0.0473) |

Primary reads with a GAIN: 5. Reads with a LOSS: 0 of 36. Reads under the null's floors: 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs.
Changed calls: L-musique musique LOSS -> WITHIN; L-hotpotqa hotpotqa GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
