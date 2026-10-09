# Full run of zlk against zrm: **NOT_ADOPTED**

R@5 of zlk's p@swa minus zrm's p@swa of the same split, on the six s1eval carves (95% question-bootstrap interval; call by docs/SCREENS.md's rule). Bold: step 1's eleven primary reads. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **-0.0022 WITHIN** | **+0.0000 WITHIN** | **+0.0065 WITHIN** | **+0.0038 WITHIN** | **+0.0032 WITHIN** | **+0.0008 WITHIN zs** |
| L-metaqa | **-0.0014 WITHIN zs** | +0.0015 WITHIN | -0.0026 WITHIN | +0.0031 WITHIN | +0.0047 WITHIN | -0.0042 WITHIN zs |
| L-squad | -0.0022 WITHIN | **-0.0013 WITHIN zs** | +0.0092 GAIN | +0.0019 WITHIN | +0.0073 WITHIN | +0.0263 GAIN zs |
| L-musique | -0.0044 WITHIN | -0.0007 WITHIN | **-0.0027 WITHIN zs** | +0.0001 WITHIN | -0.0041 WITHIN | +0.0083 WITHIN zs |
| L-hotpotqa | -0.0014 WITHIN | +0.0019 WITHIN | +0.0084 GAIN | **+0.0011 WITHIN zs** | +0.0047 WITHIN | +0.0055 WITHIN zs |
| L-2wiki | -0.0021 WITHIN | +0.0006 WITHIN | +0.0008 WITHIN | +0.0000 WITHIN | **-0.0179 LOSS zs** | +0.0095 WITHIN zs |

Primary reads with a GAIN: 0 of 11. Reads with a LOSS: 1 of 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/FULL_ROUND22.md).
