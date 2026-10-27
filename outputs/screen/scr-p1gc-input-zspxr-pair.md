P1g-c (docs/P1GC_ADOPTED_ON_P1GB.md), the input call: zspxr (zspx with R1's head before its first step) on P1g-b's pools against zspx on P1g-b's pools.

# Screen over 2 fits: **NO_GAIN**

R@5 of zspxr's p@swa minus zspx's p@swa of the same split (zspx's card fits scr-p1gc-zspx and scr-p1gc-zspx-hp, P1g-b's pools), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0035 WITHIN | -0.0012 WITHIN | -0.0079 LOSS zs | -0.0031 WITHIN | +0.0018 WITHIN | -0.0175 LOSS zs | NO_GAIN |
| L-hotpotqa | +0.0010 WITHIN | +0.0017 WITHIN | +0.0025 WITHIN | -0.0055 WITHIN zs | +0.0007 WITHIN | -0.0357 LOSS zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 3. PROMISING needs at least one GAIN and no LOSS.
