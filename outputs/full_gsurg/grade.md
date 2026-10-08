# Full run of gsurg: **NOT_ADOPTED**

R@5 of gsurg's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question-bootstrap interval; call by docs/SCREENS.md's rule). Bold: step 1's eleven primary reads. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0121 GAIN** | **-0.0018 WITHIN** | **-0.0036 WITHIN** | **+0.0019 WITHIN** | **+0.0063 WITHIN** | **+0.0008 WITHIN zs** |
| L-metaqa | **-0.0025 WITHIN zs** | +0.0008 WITHIN | -0.0015 WITHIN | -0.0007 WITHIN | +0.0097 GAIN | -0.0020 WITHIN zs |
| L-squad | +0.0026 WITHIN | **+0.0036 WITHIN zs** | -0.0094 LOSS | +0.0001 WITHIN | +0.0023 WITHIN | +0.0274 GAIN zs |
| L-musique | +0.0012 WITHIN | -0.0011 WITHIN | **+0.0842 GAIN zs** | +0.0003 WITHIN | +0.0014 WITHIN | -0.0007 WITHIN zs |
| L-hotpotqa | +0.0016 WITHIN | -0.0019 WITHIN | +0.0049 WITHIN | **-0.0147 LOSS zs** | +0.0091 GAIN | +0.0124 GAIN zs |
| L-2wiki | +0.0103 GAIN | -0.0035 WITHIN | +0.0027 WITHIN | +0.0055 WITHIN | **-0.0072 WITHIN zs** | +0.0123 GAIN zs |

Primary reads with a GAIN: 2 of 11. Reads with a LOSS: 2 of 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/FULL_BDROP20.md).
