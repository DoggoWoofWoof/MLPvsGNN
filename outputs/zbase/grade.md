# zrm's fits against zrs's (the next base): **ADOPT**

R@5 of zrm's p@swa minus zrs's p@swa of the same split, on the six s1eval carves (95% question-bootstrap interval; call by docs/SCREENS.md's rule). Bold: step 1's eleven primary reads. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0152 GAIN** | **-0.0001 WITHIN** | **+0.0026 WITHIN** | **-0.0011 WITHIN** | **+0.0046 WITHIN** | **+0.0229 GAIN zs** |
| L-metaqa | **+0.0000 WITHIN zs** | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN zs |
| L-squad | +0.0162 GAIN | **+0.0051 WITHIN zs** | +0.0040 WITHIN | +0.0014 WITHIN | +0.0013 WITHIN | +0.0242 GAIN zs |
| L-musique | +0.0153 GAIN | +0.0004 WITHIN | **+0.1334 GAIN zs** | -0.0021 WITHIN | -0.0005 WITHIN | +0.0504 GAIN zs |
| L-hotpotqa | +0.0159 GAIN | -0.0012 WITHIN | -0.0023 WITHIN | **+0.0012 WITHIN zs** | -0.0047 WITHIN | +0.0316 GAIN zs |
| L-2wiki | +0.0152 GAIN | -0.0016 WITHIN | +0.0027 WITHIN | -0.0001 WITHIN | **-0.0020 WITHIN zs** | +0.0536 GAIN zs |

Primary reads with a GAIN: 3 of 11. Reads with a LOSS: 0 of 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/BASE_ZRM_ZRS.md).
