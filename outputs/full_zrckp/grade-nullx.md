# Full run of zrckp re-graded under the seed null over every split: **ADOPT** (filed: NOT_ADOPTED)

R@5 of zrckp's p@swa minus the base arm's p@swa of the same split, on the six s1eval carves. Every read is called with its floor from the seed null over all six splits (docs/SCREENS.md, section 2, amended 8 October). Bold: step 1's eleven primary reads. zs: read zero-shot. A changed call shows filed -> re-call.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0193 GAIN (floor 0.0154)** | **+0.0005 WITHIN (floor 0.0075)** | **+0.0059 WITHIN (floor 0.0158)** | **+0.0016 WITHIN (floor 0.0075)** | **+0.0607 GAIN (floor 0.0075)** | **+0.0540 GAIN zs (floor 0.0295)** |
| L-metaqa | **+0.1077 GAIN zs (floor 0.0168)** | +0.0004 WITHIN (floor 0.0075) | +0.0097 GAIN (floor 0.0075) | +0.0073 WITHIN (floor 0.0075) | +0.0632 GAIN (floor 0.0175) | +0.0142 GAIN zs (floor 0.0075) |
| L-squad | +0.0219 GAIN (floor 0.0075) | **-0.0002 WITHIN zs (floor 0.0075)** | +0.0067 WITHIN (floor 0.0239) | +0.0045 WITHIN (floor 0.0075) | +0.0643 GAIN (floor 0.0117) | +0.0655 GAIN zs (floor 0.0260) |
| L-musique | +0.0237 GAIN (floor 0.0075) | +0.0012 WITHIN (floor 0.0075) | **-0.0114 LOSS -> WITHIN zs (floor 0.0720)** | +0.0025 WITHIN (floor 0.0075) | +0.0614 GAIN (floor 0.0075) | +0.0243 GAIN zs (floor 0.0102) |
| L-hotpotqa | +0.0251 GAIN (floor 0.0149) | +0.0015 WITHIN (floor 0.0075) | +0.0100 GAIN (floor 0.0075) | **+0.0062 WITHIN zs (floor 0.0200)** | +0.0619 GAIN (floor 0.0075) | +0.0609 GAIN zs (floor 0.0233) |
| L-2wiki | +0.0232 GAIN (floor 0.0075) | -0.0007 WITHIN (floor 0.0099) | +0.0058 WITHIN (floor 0.0075) | +0.0059 WITHIN (floor 0.0089) | **+0.0604 GAIN zs (floor 0.0112)** | +0.0644 GAIN zs (floor 0.0473) |

Primary reads with a GAIN: 5. Reads with a LOSS: 0 of 36. Reads under the null's floors: 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs.
Changed calls: L-musique musique LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
