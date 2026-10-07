# Screen over 2 fits: **MIXED**

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0176 LOSS | +0.0013 WITHIN | +0.1387 GAIN zs | +0.0004 WITHIN | -0.0041 WITHIN | +0.0175 GAIN zs | MIXED |
| L-hotpotqa | -0.0252 LOSS | -0.0003 WITHIN | +0.0115 GAIN | -0.0082 LOSS zs | -0.0025 WITHIN | -0.0172 LOSS zs | MIXED |

Reads with a GAIN: 3 of 12; with a LOSS: 4. PROMISING needs at least one GAIN and no LOSS.
