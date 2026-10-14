# Full run of zrckp against zrc: **NOT_ADOPTED**

R@5 of zrckp's p@swa minus zrc's p@swa of the same split, on the six s1eval carves (95% question-bootstrap interval; call by docs/SCREENS.md's rule). Bold: step 1's eleven primary reads. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0193 GAIN** | **+0.0005 WITHIN** | **+0.0059 WITHIN** | **+0.0016 WITHIN** | **+0.0607 GAIN** | **+0.0540 GAIN zs** |
| L-metaqa | **+0.1077 GAIN zs** | +0.0004 WITHIN | +0.0097 GAIN | +0.0073 WITHIN | +0.0632 GAIN | +0.0142 GAIN zs |
| L-squad | +0.0219 GAIN | **-0.0002 WITHIN zs** | +0.0067 WITHIN | +0.0045 WITHIN | +0.0643 GAIN | +0.0655 GAIN zs |
| L-musique | +0.0237 GAIN | +0.0012 WITHIN | **-0.0114 LOSS zs** | +0.0025 WITHIN | +0.0614 GAIN | +0.0243 GAIN zs |
| L-hotpotqa | +0.0251 GAIN | +0.0015 WITHIN | +0.0100 GAIN | **+0.0062 WITHIN zs** | +0.0619 GAIN | +0.0609 GAIN zs |
| L-2wiki | +0.0232 GAIN | -0.0007 WITHIN | +0.0058 WITHIN | +0.0059 WITHIN | **+0.0604 GAIN zs** | +0.0644 GAIN zs |

Primary reads with a GAIN: 5 of 11. Reads with a LOSS: 1 of 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/P1C3P_FULL_RUN.md).
