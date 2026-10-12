# Full run of zspx against zsp: **NOT_ADOPTED**

R@5 of zspx's p@swa minus zsp's p@swa of the same split, on the six s1eval carves (95% question-bootstrap interval; call by docs/SCREENS.md's rule). Bold: step 1's eleven primary reads. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0009 WITHIN** | **-0.0003 WITHIN** | **+0.0060 WITHIN** | **-0.0030 WITHIN** | **-0.0011 WITHIN** | **-0.0099 LOSS zs** |
| L-metaqa | **+0.0130 GAIN zs** | +0.0007 WITHIN | -0.0007 WITHIN | -0.0014 WITHIN | +0.0005 WITHIN | +0.0003 WITHIN zs |
| L-squad | +0.0030 WITHIN | **-0.0053 WITHIN zs** | +0.0016 WITHIN | +0.0036 WITHIN | +0.0046 WITHIN | -0.0187 LOSS zs |
| L-musique | +0.0018 WITHIN | -0.0010 WITHIN | **-0.0086 LOSS zs** | +0.0018 WITHIN | +0.0083 GAIN | -0.0125 LOSS zs |
| L-hotpotqa | +0.0003 WITHIN | -0.0002 WITHIN | +0.0054 WITHIN | **+0.0086 GAIN zs** | +0.0022 WITHIN | +0.0174 GAIN zs |
| L-2wiki | +0.0033 WITHIN | +0.0005 WITHIN | +0.0021 WITHIN | +0.0016 WITHIN | **+0.0023 WITHIN zs** | +0.0013 WITHIN zs |

Primary reads with a GAIN: 2 of 11. Reads with a LOSS: 4 of 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/FULL_ROUND33.md).
