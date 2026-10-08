# Full run of zrm re-graded under the seed null over every split: **ADOPT** (filed: ADOPT)

R@5 of zrm's p@swa minus the base arm's p@swa of the same split, on the six s1eval carves. Every read is called with its floor from the seed null over all six splits (docs/SCREENS.md, section 2, amended 8 October). Bold: step 1's eleven primary reads. zs: read zero-shot. A changed call shows filed -> re-call.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.1357 GAIN (floor 0.0154)** | **-0.0001 WITHIN (floor 0.0075)** | **+0.0026 WITHIN (floor 0.0158)** | **-0.0011 WITHIN (floor 0.0075)** | **+0.0046 WITHIN (floor 0.0075)** | **+0.1140 GAIN zs (floor 0.0295)** |
| L-metaqa | **+0.0000 WITHIN zs (floor 0.0168)** | +0.0000 WITHIN (floor 0.0075) | +0.0000 WITHIN (floor 0.0075) | +0.0000 WITHIN (floor 0.0075) | +0.0000 WITHIN (floor 0.0175) | +0.0000 WITHIN zs (floor 0.0075) |
| L-squad | +0.1321 GAIN (floor 0.0075) | **+0.0051 WITHIN zs (floor 0.0075)** | +0.0040 WITHIN (floor 0.0239) | +0.0014 WITHIN (floor 0.0075) | +0.0013 WITHIN (floor 0.0117) | +0.1036 GAIN zs (floor 0.0260) |
| L-musique | +0.1293 GAIN (floor 0.0075) | +0.0004 WITHIN (floor 0.0075) | **+0.1334 GAIN zs (floor 0.0720)** | -0.0021 WITHIN (floor 0.0075) | -0.0005 WITHIN (floor 0.0075) | +0.1361 GAIN zs (floor 0.0102) |
| L-hotpotqa | +0.1381 GAIN (floor 0.0149) | -0.0012 WITHIN (floor 0.0075) | -0.0023 WITHIN (floor 0.0075) | **+0.0012 WITHIN zs (floor 0.0200)** | -0.0047 WITHIN (floor 0.0075) | +0.1221 GAIN zs (floor 0.0233) |
| L-2wiki | +0.1330 GAIN (floor 0.0075) | -0.0016 WITHIN (floor 0.0099) | +0.0027 WITHIN (floor 0.0075) | -0.0001 WITHIN (floor 0.0089) | **-0.0020 WITHIN zs (floor 0.0112)** | +0.1306 GAIN zs (floor 0.0473) |

Primary reads with a GAIN: 3. Reads with a LOSS: 0 of 36. Reads under the null's floors: 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs.
Changed calls: none.
Within 0.00015 of the floor (the call stands): none.
