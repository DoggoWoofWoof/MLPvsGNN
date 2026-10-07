# Full run of rel: **NOT_ADOPTED**

R@5 of rel's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question-bootstrap interval; call by docs/SCREENS.md's rule). Bold: step 1's eleven primary reads. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0759 GAIN** | **-0.0013 WITHIN** | **-0.0013 WITHIN** | **+0.0019 WITHIN** | **+0.0001 WITHIN** | **+0.1013 GAIN zs** |
| L-metaqa | **+0.0000 WITHIN zs** | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN zs |
| L-squad | +0.0721 GAIN | **+0.0013 WITHIN zs** | -0.0093 LOSS | -0.0010 WITHIN | +0.0022 WITHIN | +0.1127 GAIN zs |
| L-musique | +0.0716 GAIN | -0.0008 WITHIN | **+0.0868 GAIN zs** | +0.0005 WITHIN | -0.0012 WITHIN | +0.0635 GAIN zs |
| L-hotpotqa | +0.0677 GAIN | -0.0012 WITHIN | +0.0053 WITHIN | **-0.0132 LOSS zs** | +0.0015 WITHIN | +0.0702 GAIN zs |
| L-2wiki | +0.0795 GAIN | -0.0025 WITHIN | -0.0033 WITHIN | +0.0059 WITHIN | **-0.0136 LOSS zs** | +0.1156 GAIN zs |

Primary reads with a GAIN: 3 of 11. Reads with a LOSS: 3 of 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/FULL_BDROP20.md).
