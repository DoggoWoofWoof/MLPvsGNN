# Full run of zrc against zrm: **ADOPT**

R@5 of zrc's p@swa minus zrm's p@swa of the same split, on the six s1eval carves (95% question-bootstrap interval; call by docs/SCREENS.md's rule). Bold: step 1's eleven primary reads. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0008 WITHIN** | **-0.0001 WITHIN** | **+0.0003 WITHIN** | **+0.0053 WITHIN** | **-0.0008 WITHIN** | **+0.0513 GAIN zs** |
| L-metaqa | **+0.0000 WITHIN zs** | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN zs |
| L-squad | +0.0007 WITHIN | **+0.0008 WITHIN zs** | +0.0020 WITHIN | -0.0001 WITHIN | -0.0001 WITHIN | +0.0406 GAIN zs |
| L-musique | -0.0007 WITHIN | -0.0006 WITHIN | **+0.0004 WITHIN zs** | +0.0001 WITHIN | +0.0024 WITHIN | +0.0257 GAIN zs |
| L-hotpotqa | -0.0008 WITHIN | -0.0003 WITHIN | +0.0045 WITHIN | **-0.0006 WITHIN zs** | +0.0051 WITHIN | +0.0450 GAIN zs |
| L-2wiki | -0.0001 WITHIN | +0.0028 WITHIN | -0.0014 WITHIN | -0.0034 WITHIN | **-0.0035 WITHIN zs** | +0.0398 GAIN zs |

Primary reads with a GAIN: 1 of 11. Reads with a LOSS: 0 of 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/FULL_ROUND18.md).
