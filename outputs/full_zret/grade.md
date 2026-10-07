# Full run of zret: **NOT_ADOPTED**

R@5 of zret's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question-bootstrap interval; call by docs/SCREENS.md's rule). Bold: step 1's eleven primary reads. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **-0.0011 WITHIN** | **+0.0013 WITHIN** | **+0.0031 WITHIN** | **-0.0005 WITHIN** | **-0.0035 WITHIN** | **+0.0250 GAIN zs** |
| L-metaqa | **+0.0644 GAIN zs** | +0.0029 WITHIN | +0.0087 GAIN | -0.0032 WITHIN | +0.0066 WITHIN | +0.0071 WITHIN zs |
| L-squad | -0.0063 WITHIN | **+0.0056 WITHIN zs** | -0.0093 LOSS | -0.0017 WITHIN | -0.0048 WITHIN | +0.0166 GAIN zs |
| L-musique | -0.0041 WITHIN | +0.0007 WITHIN | **+0.1202 GAIN zs** | +0.0046 WITHIN | -0.0018 WITHIN | -0.0012 WITHIN zs |
| L-hotpotqa | -0.0121 LOSS | +0.0011 WITHIN | +0.0073 WITHIN | **-0.0140 LOSS zs** | +0.0056 WITHIN | -0.0005 WITHIN zs |
| L-2wiki | -0.0004 WITHIN | -0.0003 WITHIN | +0.0055 WITHIN | +0.0058 WITHIN | **-0.0081 LOSS zs** | +0.0262 GAIN zs |

Primary reads with a GAIN: 3 of 11. Reads with a LOSS: 4 of 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/FULL_BDROP20.md).
