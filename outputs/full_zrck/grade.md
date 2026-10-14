# Full run of zrck against zrc: **NOT_ADOPTED**

R@5 of zrck's p@swa minus zrc's p@swa of the same split, on the six s1eval carves (95% question-bootstrap interval; call by docs/SCREENS.md's rule). Bold: step 1's eleven primary reads. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0196 GAIN** | **+0.0001 WITHIN** | **+0.0019 WITHIN** | **-0.0012 WITHIN** | **+0.0573 GAIN** | **+0.0529 GAIN zs** |
| L-metaqa | **+0.0619 GAIN zs** | +0.0003 WITHIN | +0.0042 WITHIN | +0.0063 WITHIN | +0.0662 GAIN | +0.0212 GAIN zs |
| L-squad | +0.0216 GAIN | **-0.0028 WITHIN zs** | +0.0061 WITHIN | +0.0034 WITHIN | +0.0623 GAIN | +0.0721 GAIN zs |
| L-musique | +0.0225 GAIN | +0.0013 WITHIN | **-0.0176 LOSS zs** | +0.0022 WITHIN | +0.0632 GAIN | +0.0372 GAIN zs |
| L-hotpotqa | +0.0259 GAIN | +0.0009 WITHIN | +0.0050 WITHIN | **+0.0113 GAIN zs** | +0.0397 GAIN | +0.0820 GAIN zs |
| L-2wiki | +0.0215 GAIN | -0.0019 WITHIN | +0.0039 WITHIN | +0.0045 WITHIN | **+0.0639 GAIN zs** | +0.0548 GAIN zs |

Primary reads with a GAIN: 6 of 11. Reads with a LOSS: 1 of 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/P1C3_FULL_RUN.md).
