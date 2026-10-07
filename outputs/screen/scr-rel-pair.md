# Screen over 2 fits: **MIXED**

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0716 GAIN | -0.0008 WITHIN | +0.0868 GAIN zs | +0.0005 WITHIN | -0.0012 WITHIN | +0.0635 GAIN zs | PROMISING |
| L-hotpotqa | +0.0677 GAIN | -0.0012 WITHIN | +0.0053 WITHIN | -0.0132 LOSS zs | +0.0015 WITHIN | +0.0702 GAIN zs | MIXED |

Reads with a GAIN: 5 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
