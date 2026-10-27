A5cp (docs/A5CP_P1GB_POOLS.md): zrckpc (zrckp with zck's coupled kind offsets on typed graphs (zret's output layer and base weight, trained jointly)) against zrckp, both on P1g-b's pools.

# Screen over 2 fits: **MIXED**

R@5 of zrckpc's p@swa minus zrckp's p@swa of the same split (zrckp's card fits scr-p1gb-zrckp and scr-p1gb-zrckp-hp, P1g-b's pools), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0004 WITHIN | -0.0009 WITHIN | +0.0098 GAIN zs | +0.0026 WITHIN | -0.0045 WITHIN | +0.0160 GAIN zs | PROMISING |
| L-hotpotqa | +0.0022 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | -0.0019 WITHIN zs | -0.0084 LOSS | +0.0366 GAIN zs | MIXED |

Reads with a GAIN: 3 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
