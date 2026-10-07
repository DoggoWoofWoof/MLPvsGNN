# Screen over 2 fits: **MIXED**

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0006 WITHIN | +0.0003 WITHIN | +0.0219 GAIN zs | -0.0028 WITHIN | +0.0035 WITHIN | -0.0090 WITHIN zs | PROMISING |
| L-hotpotqa | -0.0064 WITHIN | -0.0013 WITHIN | -0.0038 WITHIN | -0.0253 LOSS zs | +0.0037 WITHIN | -0.0042 WITHIN zs | NO_GAIN |

Reads with a GAIN: 1 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
