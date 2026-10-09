# Full run of zkind against zrm: **NOT_ADOPTED**

R@5 of zkind's p@swa minus zrm's p@swa of the same split, on the six s1eval carves (95% question-bootstrap interval; call by docs/SCREENS.md's rule). Bold: step 1's eleven primary reads. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0008 WITHIN** | **+0.0007 WITHIN** | **+0.0034 WITHIN** | **-0.0011 WITHIN** | **-0.0046 WITHIN** | **+0.0372 GAIN zs** |
| L-metaqa | **+0.0000 WITHIN zs** | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN zs |
| L-squad | -0.0006 WITHIN | **-0.0067 WITHIN zs** | +0.0009 WITHIN | +0.0018 WITHIN | +0.0017 WITHIN | +0.0461 GAIN zs |
| L-musique | -0.0003 WITHIN | -0.0001 WITHIN | **+0.0100 GAIN zs** | -0.0012 WITHIN | -0.0022 WITHIN | +0.0168 GAIN zs |
| L-hotpotqa | -0.0022 WITHIN | +0.0024 WITHIN | +0.0078 GAIN | **-0.0113 LOSS zs** | +0.0007 WITHIN | +0.0591 GAIN zs |
| L-2wiki | +0.0001 WITHIN | +0.0024 WITHIN | +0.0003 WITHIN | -0.0020 WITHIN | **-0.0010 WITHIN zs** | +0.0328 GAIN zs |

Primary reads with a GAIN: 2 of 11. Reads with a LOSS: 1 of 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/FULL_ROUND19.md).
