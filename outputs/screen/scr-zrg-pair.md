# Screen over 2 fits: **MIXED**

R@5 of zrg's p@swa minus zrm's p@swa of the same split (zrm's screen fits scr-zrm and scr-zrm-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0012 WITHIN | +0.0016 WITHIN | -0.0040 WITHIN zs | +0.0022 WITHIN | +0.0027 WITHIN | +0.0018 WITHIN zs | NO_GAIN |
| L-hotpotqa | +0.0003 WITHIN | +0.0008 WITHIN | +0.0087 GAIN | -0.0094 LOSS zs | +0.0053 WITHIN | +0.0081 WITHIN zs | MIXED |

Reads with a GAIN: 1 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
