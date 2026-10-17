R1 (docs/R1_RESIDUAL_INPUT.md): zrckpr (zrckp plus the R1 head) against zrckp.

# Screen over 2 fits: **PROMISING**

R@5 of zrckpr's p@swa minus zrckp's p@swa of the same split (zrckp's card fits scr-p1c3p-zrckp and scr-p1c3p-zrckp-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0007 WITHIN | +0.0011 WITHIN | +0.0076 GAIN zs | +0.0018 WITHIN | +0.0014 WITHIN | +0.0135 GAIN zs | PROMISING |
| L-hotpotqa | -0.0004 WITHIN | -0.0013 WITHIN | +0.0034 WITHIN | +0.0045 WITHIN zs | -0.0019 WITHIN | -0.0064 WITHIN zs | NO_GAIN |

Reads with a GAIN: 2 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
