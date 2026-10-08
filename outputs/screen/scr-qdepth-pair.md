# Screen over 2 fits: **MIXED**

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0014 WITHIN | +0.0001 WITHIN | +0.1106 GAIN zs | -0.0038 WITHIN | -0.0054 WITHIN | -0.0204 LOSS zs | MIXED |
| L-hotpotqa | -0.0034 WITHIN | +0.0013 WITHIN | +0.0035 WITHIN | -0.0125 LOSS zs | +0.0035 WITHIN | -0.0035 WITHIN zs | NO_GAIN |

Reads with a GAIN: 1 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
