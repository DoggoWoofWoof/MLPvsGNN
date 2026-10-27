A7p (docs/A7P_P1GB_POOLS.md): zrckps (zrckp plus the A7 head) against zrckp, both on P1g-b's pools.

# Screen over 2 fits: **PROMISING**

R@5 of zrckps's p@swa minus zrckp's p@swa of the same split (zrckp's card fits scr-p1gb-zrckp and scr-p1gb-zrckp-hp, P1g-b's pools), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0022 WITHIN | -0.0001 WITHIN | +0.0114 GAIN zs | +0.0051 WITHIN | -0.0004 WITHIN | +0.0194 GAIN zs | PROMISING |
| L-hotpotqa | +0.0044 WITHIN | +0.0017 WITHIN | +0.0003 WITHIN | +0.0076 GAIN zs | -0.0007 WITHIN | +0.0273 GAIN zs | PROMISING |

Reads with a GAIN: 4 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
