# Screen over 2 fits: **PROMISING**

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.1268 GAIN | +0.0007 WITHIN | +0.2324 GAIN zs | +0.0023 WITHIN | +0.0003 WITHIN | +0.1376 GAIN zs | PROMISING |
| L-hotpotqa | +0.1250 GAIN | -0.0015 WITHIN | +0.0038 WITHIN | -0.0072 WITHIN zs | +0.0051 WITHIN | +0.1446 GAIN zs | PROMISING |

Reads with a GAIN: 5 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
