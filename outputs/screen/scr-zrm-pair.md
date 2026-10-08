# Screen over 2 fits: **PROMISING**

R@5 of zrm's p@swa minus zret's p@swa of the same split (zret's fits: its screen fit scr-zret and its full run's L-hotpotqa fit), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.1293 GAIN | +0.0004 WITHIN | +0.1334 GAIN zs | -0.0021 WITHIN | -0.0005 WITHIN | +0.1361 GAIN zs | PROMISING |
| L-hotpotqa | +0.1381 GAIN | -0.0012 WITHIN | -0.0023 WITHIN | +0.0012 WITHIN zs | -0.0047 WITHIN | +0.1221 GAIN zs | PROMISING |

Reads with a GAIN: 5 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
