A3vp (docs/A3VP_P1GB_POOLS.md): zrckpy (zrckp trained with A3v's two-sided share augmentation of zret's reference) against zrckp, both on P1g-b's pools.

# Screen over 2 fits: **NO_GAIN**

R@5 of zrckpy's p@swa minus zrckp's p@swa of the same split (zrckp's card fits scr-p1gb-zrckp and scr-p1gb-zrckp-hp, P1g-b's pools), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0012 WITHIN | -0.0010 WITHIN | +0.0054 WITHIN zs | +0.0007 WITHIN | -0.0069 WITHIN | -0.0113 LOSS zs | NO_GAIN |
| L-hotpotqa | +0.0030 WITHIN | +0.0003 WITHIN | -0.0058 WITHIN | -0.0037 WITHIN zs | -0.0054 WITHIN | -0.0181 LOSS zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
