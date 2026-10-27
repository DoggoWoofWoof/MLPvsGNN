P5c5 (docs/P5C5_FIVE_FOLDS.md): zrckpo5 (zrckp plus P5c's head over the four columns, g cross-fitted over five folds) against zrckp.

# Screen over 2 fits: **NO_GAIN**

R@5 of zrckpo5's p@swa minus zrckp's p@swa of the same split (zrckp's card fits scr-p1c3p-zrckp and scr-p1c3p-zrckp-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0006 WITHIN | -0.0003 WITHIN | +0.0011 WITHIN zs | +0.0003 WITHIN | +0.0018 WITHIN | -0.0076 WITHIN zs | NO_GAIN |
| L-hotpotqa | -0.0008 WITHIN | -0.0007 WITHIN | +0.0030 WITHIN | +0.0031 WITHIN zs | -0.0004 WITHIN | -0.0120 LOSS zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
