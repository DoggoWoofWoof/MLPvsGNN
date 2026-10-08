# Full run of gsurg re-called under the seed null: **NOT_ADOPTED** (filed: NOT_ADOPTED)

The grade's reads on L-musique and L-hotpotqa (the screen's reused fits) are called with the seed null's floors (docs/SCREENS.md, section 2); the other splits keep 0.0075, since no null covers them. Bold: step 1's eleven primary reads. zs: read zero-shot. A changed call shows filed -> re-call.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0121 GAIN** | **-0.0018 WITHIN** | **-0.0036 WITHIN** | **+0.0019 WITHIN** | **+0.0063 WITHIN** | **+0.0008 WITHIN zs** |
| L-metaqa | **-0.0025 WITHIN zs** | +0.0008 WITHIN | -0.0015 WITHIN | -0.0007 WITHIN | +0.0097 GAIN | -0.0020 WITHIN zs |
| L-squad | +0.0026 WITHIN | **+0.0036 WITHIN zs** | -0.0094 LOSS | +0.0001 WITHIN | +0.0023 WITHIN | +0.0274 GAIN zs |
| L-musique | +0.0012 WITHIN (floor 0.0075) | -0.0011 WITHIN (floor 0.0075) | **+0.0842 GAIN zs (floor 0.0720)** | +0.0003 WITHIN (floor 0.0075) | +0.0014 WITHIN (floor 0.0075) | -0.0007 WITHIN zs (floor 0.0102) |
| L-hotpotqa | +0.0016 WITHIN (floor 0.0149) | -0.0019 WITHIN (floor 0.0075) | +0.0049 WITHIN (floor 0.0075) | **-0.0147 LOSS -> WITHIN zs (floor 0.0200)** | +0.0091 GAIN (floor 0.0075) | +0.0124 GAIN -> WITHIN zs (floor 0.0233) |
| L-2wiki | +0.0103 GAIN | -0.0035 WITHIN | +0.0027 WITHIN | +0.0055 WITHIN | **-0.0072 WITHIN zs** | +0.0123 GAIN zs |

Primary reads with a GAIN: 2. Reads with a LOSS: 1 of 36. Reads under the null's floors: 12; under 0.0075: 24.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/FULL_BDROP20.md).
Changed calls: L-hotpotqa hotpotqa LOSS -> WITHIN; L-hotpotqa webqsp GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
