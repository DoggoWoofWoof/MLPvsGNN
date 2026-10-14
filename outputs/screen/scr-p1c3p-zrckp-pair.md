P1c3p (docs/P1C3P_POINTWISE_HEAD.md): zrckp (the pointwise KT head) on KIND_T's pools against zrc on today's.

# Screen over 2 fits: **MIXED**

R@5 of zrckp's p@swa minus zrc's p@swa of the same split (zrc's card fits scr-zrct and scr-zrct-hp, today's pools), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0237 GAIN | +0.0012 WITHIN | -0.0114 LOSS zs | +0.0025 WITHIN | +0.0614 GAIN | +0.0243 GAIN zs | MIXED |
| L-hotpotqa | +0.0251 GAIN | +0.0015 WITHIN | +0.0100 GAIN | +0.0062 WITHIN zs | +0.0619 GAIN | +0.0609 GAIN zs | PROMISING |

Reads with a GAIN: 7 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
