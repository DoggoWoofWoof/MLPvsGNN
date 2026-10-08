# Full run of zsp against zrm: **ADOPT**

R@5 of zsp's p@swa minus zrm's p@swa of the same split, on the six s1eval carves (95% question-bootstrap interval; call by docs/SCREENS.md's rule). Bold: step 1's eleven primary reads. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0016 WITHIN** | **-0.0007 WITHIN** | **-0.0073 WITHIN** | **+0.0026 WITHIN** | **+0.0041 WITHIN** | **+0.0268 GAIN zs** |
| L-metaqa | **+0.0303 GAIN zs** | +0.0016 WITHIN | -0.0007 WITHIN | +0.0026 WITHIN | +0.0105 GAIN | +0.0092 GAIN zs |
| L-squad | +0.0003 WITHIN | **+0.0008 WITHIN zs** | +0.0012 WITHIN | +0.0032 WITHIN | -0.0007 WITHIN | +0.0154 GAIN zs |
| L-musique | +0.0006 WITHIN | -0.0018 WITHIN | **+0.0004 WITHIN zs** | +0.0013 WITHIN | +0.0052 WITHIN | +0.0035 WITHIN zs |
| L-hotpotqa | +0.0013 WITHIN | +0.0035 WITHIN | +0.0034 WITHIN | **-0.0074 WITHIN zs** | +0.0100 GAIN | -0.0030 WITHIN zs |
| L-2wiki | +0.0011 WITHIN | +0.0000 WITHIN | +0.0028 WITHIN | +0.0099 GAIN | **+0.0141 GAIN zs** | +0.0163 GAIN zs |

Primary reads with a GAIN: 3 of 11. Reads with a LOSS: 0 of 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/FULL_ROUND23.md).
