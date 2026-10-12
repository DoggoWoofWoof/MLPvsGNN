# Full run of zep against zrc: **ADOPT**

R@5 of zep's p@swa minus zrc's p@swa of the same split, on the six s1eval carves (95% question-bootstrap interval; call by docs/SCREENS.md's rule). Bold: step 1's eleven primary reads. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0020 WITHIN** | **+0.0008 WITHIN** | **+0.0043 WITHIN** | **-0.0017 WITHIN** | **+0.0028 WITHIN** | **+0.0204 GAIN zs** |
| L-metaqa | **+0.0121 GAIN zs** | +0.0008 WITHIN | -0.0027 WITHIN | +0.0007 WITHIN | +0.0018 WITHIN | +0.0020 WITHIN zs |
| L-squad | +0.0019 WITHIN | **-0.0002 WITHIN zs** | +0.0002 WITHIN | +0.0028 WITHIN | +0.0041 WITHIN | +0.0145 GAIN zs |
| L-musique | +0.0021 WITHIN | +0.0022 WITHIN | **-0.0068 WITHIN zs** | +0.0018 WITHIN | +0.0017 WITHIN | +0.0118 GAIN zs |
| L-hotpotqa | +0.0026 WITHIN | +0.0010 WITHIN | -0.0066 WITHIN | **-0.0021 WITHIN zs** | +0.0004 WITHIN | +0.0228 GAIN zs |
| L-2wiki | +0.0027 WITHIN | -0.0001 WITHIN | -0.0038 WITHIN | +0.0052 WITHIN | **+0.0046 WITHIN zs** | +0.0122 GAIN zs |

Primary reads with a GAIN: 2 of 11. Reads with a LOSS: 0 of 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/FULL_ROUND33.md).
