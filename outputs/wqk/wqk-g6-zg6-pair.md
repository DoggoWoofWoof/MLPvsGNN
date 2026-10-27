# Screen over 2 fits: **MIXED**

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| WQ-musique | +0.0409 GAIN zs | -0.0013 WITHIN | +0.0428 GAIN zs | +0.0051 WITHIN | +0.0045 WITHIN | +0.0089 WITHIN | PROMISING |
| WQ-hotpotqa | +0.0783 GAIN zs | +0.0000 WITHIN | +0.0167 GAIN | -0.0313 LOSS zs | +0.0084 GAIN | +0.0268 GAIN | MIXED |

Reads with a GAIN: 6 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
