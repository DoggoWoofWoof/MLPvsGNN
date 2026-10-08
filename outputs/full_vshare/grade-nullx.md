# Full run of vshare re-graded under the seed null over every split: **NOT_ADOPTED** (filed: NOT_ADOPTED)

R@5 of vshare's p@swa minus step 1's p@swa of the same split, on the six s1eval carves. Every read is called with its floor from the seed null over all six splits (docs/SCREENS.md, section 2, amended 8 October). Bold: step 1's eleven primary reads. zs: read zero-shot. A changed call shows filed -> re-call.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0023 WITHIN (floor 0.0154)** | **-0.0031 WITHIN (floor 0.0075)** | **-0.0108 LOSS -> WITHIN (floor 0.0158)** | **-0.0030 WITHIN (floor 0.0075)** | **-0.0010 WITHIN (floor 0.0075)** | **-0.0070 WITHIN zs (floor 0.0295)** |
| L-metaqa | **+0.0180 GAIN zs (floor 0.0168)** | +0.0026 WITHIN (floor 0.0075) | -0.0018 WITHIN (floor 0.0075) | -0.0056 WITHIN (floor 0.0075) | +0.0067 WITHIN (floor 0.0175) | -0.0030 WITHIN zs (floor 0.0075) |
| L-squad | -0.0094 LOSS (floor 0.0075) | **+0.0120 GAIN zs (floor 0.0075)** | -0.0127 LOSS -> WITHIN (floor 0.0239) | -0.0049 WITHIN (floor 0.0075) | -0.0013 WITHIN (floor 0.0117) | +0.0008 WITHIN zs (floor 0.0260) |
| L-musique | -0.0030 WITHIN (floor 0.0075) | -0.0004 WITHIN (floor 0.0075) | **+0.0891 GAIN zs (floor 0.0720)** | -0.0017 WITHIN (floor 0.0075) | -0.0011 WITHIN (floor 0.0075) | +0.0045 WITHIN zs (floor 0.0102) |
| L-hotpotqa | -0.0055 WITHIN (floor 0.0149) | +0.0003 WITHIN (floor 0.0075) | +0.0030 WITHIN (floor 0.0075) | **-0.0140 LOSS -> WITHIN zs (floor 0.0200)** | -0.0013 WITHIN (floor 0.0075) | -0.0057 WITHIN zs (floor 0.0233) |
| L-2wiki | +0.0043 WITHIN (floor 0.0075) | -0.0022 WITHIN (floor 0.0099) | -0.0047 WITHIN (floor 0.0075) | +0.0030 WITHIN (floor 0.0089) | **-0.0021 WITHIN zs (floor 0.0112)** | +0.0127 GAIN -> WITHIN zs (floor 0.0473) |

Primary reads with a GAIN: 3. Reads with a LOSS: 1 of 36. Reads under the null's floors: 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs.
Changed calls: J5 musique LOSS -> WITHIN; L-squad musique LOSS -> WITHIN; L-hotpotqa hotpotqa LOSS -> WITHIN; L-2wiki webqsp GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
