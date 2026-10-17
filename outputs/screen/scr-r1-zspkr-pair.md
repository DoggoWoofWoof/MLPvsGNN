R1 (docs/R1_RESIDUAL_INPUT.md): zspkr (zspk plus the R1 head before its propagation) against zspk.

# Screen over 2 fits: **NO_GAIN**

R@5 of zspkr's p@swa minus zspk's p@swa of the same split (zspk's card fits scr-p1c3-zspk and scr-p1c3-zspk-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0017 WITHIN | +0.0021 WITHIN | -0.0079 LOSS zs | -0.0005 WITHIN | -0.0041 WITHIN | -0.0005 WITHIN zs | NO_GAIN |
| L-hotpotqa | -0.0016 WITHIN | -0.0003 WITHIN | +0.0011 WITHIN | -0.0032 WITHIN zs | -0.0009 WITHIN | +0.0078 WITHIN zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
