# Full run of rel re-called under the seed null: **NOT_ADOPTED** (filed: NOT_ADOPTED)

The grade's reads on L-musique and L-hotpotqa (the screen's reused fits) are called with the seed null's floors (docs/SCREENS.md, section 2); the other splits keep 0.0075, since no null covers them. Bold: step 1's eleven primary reads. zs: read zero-shot. A changed call shows filed -> re-call.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0759 GAIN** | **-0.0013 WITHIN** | **-0.0013 WITHIN** | **+0.0019 WITHIN** | **+0.0001 WITHIN** | **+0.1013 GAIN zs** |
| L-metaqa | **+0.0000 WITHIN zs** | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN zs |
| L-squad | +0.0721 GAIN | **+0.0013 WITHIN zs** | -0.0093 LOSS | -0.0010 WITHIN | +0.0022 WITHIN | +0.1127 GAIN zs |
| L-musique | +0.0716 GAIN (floor 0.0075) | -0.0008 WITHIN (floor 0.0075) | **+0.0868 GAIN zs (floor 0.0720)** | +0.0005 WITHIN (floor 0.0075) | -0.0012 WITHIN (floor 0.0075) | +0.0635 GAIN zs (floor 0.0102) |
| L-hotpotqa | +0.0677 GAIN (floor 0.0149) | -0.0012 WITHIN (floor 0.0075) | +0.0053 WITHIN (floor 0.0075) | **-0.0132 LOSS -> WITHIN zs (floor 0.0200)** | +0.0015 WITHIN (floor 0.0075) | +0.0702 GAIN zs (floor 0.0233) |
| L-2wiki | +0.0795 GAIN | -0.0025 WITHIN | -0.0033 WITHIN | +0.0059 WITHIN | **-0.0136 LOSS zs** | +0.1156 GAIN zs |

Primary reads with a GAIN: 3. Reads with a LOSS: 2 of 36. Reads under the null's floors: 12; under 0.0075: 24.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/FULL_BDROP20.md).
Changed calls: L-hotpotqa hotpotqa LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
