# Full run of zrm against zret: **ADOPT**

R@5 of zrm's p@swa minus zret's p@swa of the same split, on the six s1eval carves (95% question-bootstrap interval; call by docs/SCREENS.md's rule). Bold: step 1's eleven primary reads. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.1357 GAIN** | **-0.0001 WITHIN** | **+0.0026 WITHIN** | **-0.0011 WITHIN** | **+0.0046 WITHIN** | **+0.1140 GAIN zs** |
| L-metaqa | **+0.0000 WITHIN zs** | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN zs |
| L-squad | +0.1321 GAIN | **+0.0051 WITHIN zs** | +0.0040 WITHIN | +0.0014 WITHIN | +0.0013 WITHIN | +0.1036 GAIN zs |
| L-musique | +0.1293 GAIN | +0.0004 WITHIN | **+0.1334 GAIN zs** | -0.0021 WITHIN | -0.0005 WITHIN | +0.1361 GAIN zs |
| L-hotpotqa | +0.1381 GAIN | -0.0012 WITHIN | -0.0023 WITHIN | **+0.0012 WITHIN zs** | -0.0047 WITHIN | +0.1221 GAIN zs |
| L-2wiki | +0.1330 GAIN | -0.0016 WITHIN | +0.0027 WITHIN | -0.0001 WITHIN | **-0.0020 WITHIN zs** | +0.1306 GAIN zs |

Primary reads with a GAIN: 3 of 11. Reads with a LOSS: 0 of 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/FULL_ROUND14.md).
