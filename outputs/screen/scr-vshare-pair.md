# Screen over 2 fits: **MIXED**

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0030 WITHIN | -0.0004 WITHIN | +0.0891 GAIN zs | -0.0017 WITHIN | -0.0011 WITHIN | +0.0045 WITHIN zs | PROMISING |
| L-hotpotqa | -0.0055 WITHIN | +0.0003 WITHIN | +0.0030 WITHIN | -0.0140 LOSS zs | -0.0013 WITHIN | -0.0057 WITHIN zs | NO_GAIN |

Reads with a GAIN: 1 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
