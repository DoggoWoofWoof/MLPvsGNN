A10e (docs/A10E_G2_END_TO_END.md): zg2xe (zspx with G2's hidden-state layers over its score, trained end to end) against zspx.

# Screen over 2 fits: **NO_GAIN**

R@5 of zg2xe's p@swa minus zspx's p@swa of the same split (zspx's card fits scr-g5-zspx and scr-g5-zspx-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0001 WITHIN | +0.0024 WITHIN | -0.0169 LOSS zs | +0.0011 WITHIN | -0.0054 WITHIN | -0.0037 WITHIN zs | NO_GAIN |
| L-hotpotqa | -0.0012 WITHIN | +0.0008 WITHIN | -0.0041 WITHIN | -0.0010 WITHIN zs | +0.0034 WITHIN | -0.0080 WITHIN zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
