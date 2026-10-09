# Screen over 2 fits: **MIXED**

R@5 of zg1's p@swa minus zsp's p@swa of the same split (zsp's card fits scr-zspg and scr-zspg-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0011 WITHIN | +0.0003 WITHIN | +0.0077 GAIN zs | +0.0005 WITHIN | +0.0062 WITHIN | -0.0156 LOSS zs | MIXED |
| L-hotpotqa | -0.0024 WITHIN | +0.0010 WITHIN | +0.0030 WITHIN | +0.0050 WITHIN zs | +0.0072 WITHIN | +0.0023 WITHIN zs | NO_GAIN |

Reads with a GAIN: 1 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
