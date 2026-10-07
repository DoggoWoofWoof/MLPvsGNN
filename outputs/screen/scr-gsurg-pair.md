# Screen over 2 fits: **MIXED**

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0012 WITHIN | -0.0011 WITHIN | +0.0842 GAIN zs | +0.0003 WITHIN | +0.0014 WITHIN | -0.0007 WITHIN zs | PROMISING |
| L-hotpotqa | +0.0016 WITHIN | -0.0019 WITHIN | +0.0049 WITHIN | -0.0147 LOSS zs | +0.0091 GAIN | +0.0124 GAIN zs | MIXED |

Reads with a GAIN: 3 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
