# Full run of rmatch: **NOT_ADOPTED**

R@5 of rmatch's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question-bootstrap interval; call by docs/SCREENS.md's rule). Bold: step 1's eleven primary reads. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.1345 GAIN** | **-0.0020 WITHIN** | **+0.0077 GAIN** | **+0.0014 WITHIN** | **+0.0036 WITHIN** | **+0.1628 GAIN zs** |
| L-metaqa | **+0.0000 WITHIN zs** | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN zs |
| L-squad | +0.1251 GAIN | **+0.0096 GAIN zs** | -0.0073 WITHIN | -0.0037 WITHIN | -0.0029 WITHIN | +0.1409 GAIN zs |
| L-musique | +0.1268 GAIN | +0.0007 WITHIN | **+0.2324 GAIN zs** | +0.0023 WITHIN | +0.0003 WITHIN | +0.1376 GAIN zs |
| L-hotpotqa | +0.1250 GAIN | -0.0015 WITHIN | +0.0038 WITHIN | **-0.0072 WITHIN zs** | +0.0051 WITHIN | +0.1446 GAIN zs |
| L-2wiki | +0.1334 GAIN | -0.0013 WITHIN | -0.0025 WITHIN | +0.0029 WITHIN | **-0.0136 LOSS zs** | +0.1532 GAIN zs |

Primary reads with a GAIN: 5 of 11. Reads with a LOSS: 1 of 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/FULL_BDROP20.md).
