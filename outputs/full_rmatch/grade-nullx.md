# Full run of rmatch re-graded under the seed null over every split: **NOT_ADOPTED** (filed: NOT_ADOPTED)

R@5 of rmatch's p@swa minus step 1's p@swa of the same split, on the six s1eval carves. Every read is called with its floor from the seed null over all six splits (docs/SCREENS.md, section 2, amended 8 October). Bold: step 1's eleven primary reads. zs: read zero-shot. A changed call shows filed -> re-call.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.1345 GAIN (floor 0.0154)** | **-0.0020 WITHIN (floor 0.0075)** | **+0.0077 GAIN -> WITHIN (floor 0.0158)** | **+0.0014 WITHIN (floor 0.0075)** | **+0.0036 WITHIN (floor 0.0075)** | **+0.1628 GAIN zs (floor 0.0295)** |
| L-metaqa | **+0.0000 WITHIN zs (floor 0.0168)** | +0.0000 WITHIN (floor 0.0075) | +0.0000 WITHIN (floor 0.0075) | +0.0000 WITHIN (floor 0.0075) | +0.0000 WITHIN (floor 0.0175) | +0.0000 WITHIN zs (floor 0.0075) |
| L-squad | +0.1251 GAIN (floor 0.0075) | **+0.0096 GAIN zs (floor 0.0075)** | -0.0073 WITHIN (floor 0.0239) | -0.0037 WITHIN (floor 0.0075) | -0.0029 WITHIN (floor 0.0117) | +0.1409 GAIN zs (floor 0.0260) |
| L-musique | +0.1268 GAIN (floor 0.0075) | +0.0007 WITHIN (floor 0.0075) | **+0.2324 GAIN zs (floor 0.0720)** | +0.0023 WITHIN (floor 0.0075) | +0.0003 WITHIN (floor 0.0075) | +0.1376 GAIN zs (floor 0.0102) |
| L-hotpotqa | +0.1250 GAIN (floor 0.0149) | -0.0015 WITHIN (floor 0.0075) | +0.0038 WITHIN (floor 0.0075) | **-0.0072 WITHIN zs (floor 0.0200)** | +0.0051 WITHIN (floor 0.0075) | +0.1446 GAIN zs (floor 0.0233) |
| L-2wiki | +0.1334 GAIN (floor 0.0075) | -0.0013 WITHIN (floor 0.0099) | -0.0025 WITHIN (floor 0.0075) | +0.0029 WITHIN (floor 0.0089) | **-0.0136 LOSS zs (floor 0.0112)** | +0.1532 GAIN zs (floor 0.0473) |

Primary reads with a GAIN: 4. Reads with a LOSS: 1 of 36. Reads under the null's floors: 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs.
Changed calls: J5 musique GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
