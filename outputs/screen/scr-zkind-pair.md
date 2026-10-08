# Screen over 2 fits: **MIXED**

R@5 of zkind's p@swa minus zrm's p@swa of the same split (zrm's screen fits scr-zrm and scr-zrm-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0003 WITHIN | -0.0001 WITHIN | +0.0100 GAIN zs | -0.0012 WITHIN | -0.0022 WITHIN | +0.0168 GAIN zs | PROMISING |
| L-hotpotqa | -0.0022 WITHIN | +0.0024 WITHIN | +0.0078 GAIN | -0.0113 LOSS zs | +0.0007 WITHIN | +0.0591 GAIN zs | MIXED |

Reads with a GAIN: 4 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
