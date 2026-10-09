# Full run of zrg against zrm: **NOT_ADOPTED**

R@5 of zrg's p@swa minus zrm's p@swa of the same split, on the six s1eval carves (95% question-bootstrap interval; call by docs/SCREENS.md's rule). Bold: step 1's eleven primary reads. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0017 WITHIN** | **-0.0002 WITHIN** | **+0.0058 WITHIN** | **+0.0051 WITHIN** | **+0.0026 WITHIN** | **+0.0138 GAIN zs** |
| L-metaqa | **+0.0027 WITHIN zs** | +0.0001 WITHIN | -0.0045 WITHIN | +0.0018 WITHIN | +0.0012 WITHIN | +0.0058 WITHIN zs |
| L-squad | +0.0008 WITHIN | **-0.0032 WITHIN zs** | -0.0001 WITHIN | +0.0028 WITHIN | +0.0034 WITHIN | +0.0273 GAIN zs |
| L-musique | +0.0012 WITHIN | +0.0016 WITHIN | **-0.0040 WITHIN zs** | +0.0022 WITHIN | +0.0027 WITHIN | +0.0018 WITHIN zs |
| L-hotpotqa | +0.0003 WITHIN | +0.0008 WITHIN | +0.0087 GAIN | **-0.0094 LOSS zs** | +0.0053 WITHIN | +0.0081 WITHIN zs |
| L-2wiki | +0.0014 WITHIN | +0.0008 WITHIN | -0.0060 WITHIN | +0.0021 WITHIN | **-0.0068 WITHIN zs** | +0.0041 WITHIN zs |

Primary reads with a GAIN: 1 of 11. Reads with a LOSS: 1 of 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/FULL_ROUND21.md).
