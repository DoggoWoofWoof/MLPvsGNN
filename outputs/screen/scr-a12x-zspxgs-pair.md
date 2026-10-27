A12x (docs/A12X_GSURG_ZSPX.md): zspxgs (zspx trained with gsurg's loop) against zspx.

# Screen over 2 fits: **NO_GAIN**

R@5 of zspxgs's p@swa minus zspx's p@swa of the same split (zspx's card fits scr-g5-zspx and scr-g5-zspx-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0012 WITHIN | -0.0003 WITHIN | -0.0133 LOSS zs | +0.0027 WITHIN | -0.0010 WITHIN | -0.0004 WITHIN zs | NO_GAIN |
| L-hotpotqa | -0.0014 WITHIN | -0.0001 WITHIN | -0.0050 WITHIN | -0.0033 WITHIN zs | +0.0051 WITHIN | -0.0094 LOSS zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
