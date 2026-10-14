# Full run of zspk against zsp: **NOT_ADOPTED**

R@5 of zspk's p@swa minus zsp's p@swa of the same split, on the six s1eval carves (95% question-bootstrap interval; call by docs/SCREENS.md's rule). Bold: step 1's eleven primary reads. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0211 GAIN** | **-0.0009 WITHIN** | **+0.0110 GAIN** | **-0.0011 WITHIN** | **+0.0586 GAIN** | **+0.0331 GAIN zs** |
| L-metaqa | **+0.0422 GAIN zs** | -0.0029 WITHIN | +0.0073 WITHIN | +0.0010 WITHIN | +0.0563 GAIN | +0.0021 WITHIN zs |
| L-squad | +0.0255 GAIN | **+0.0015 WITHIN zs** | +0.0075 WITHIN | +0.0028 WITHIN | +0.0533 GAIN | +0.0263 GAIN zs |
| L-musique | +0.0215 GAIN | -0.0011 WITHIN | **-0.0100 LOSS zs** | +0.0015 WITHIN | +0.0659 GAIN | -0.0017 WITHIN zs |
| L-hotpotqa | +0.0233 GAIN | +0.0026 WITHIN | +0.0107 GAIN | **+0.0064 WITHIN zs** | +0.0589 GAIN | +0.0418 GAIN zs |
| L-2wiki | +0.0247 GAIN | +0.0004 WITHIN | +0.0070 WITHIN | +0.0032 WITHIN | **+0.0663 GAIN zs** | +0.0351 GAIN zs |

Primary reads with a GAIN: 6 of 11. Reads with a LOSS: 1 of 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/P1C3_FULL_RUN.md).
