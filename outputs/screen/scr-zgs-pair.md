# Screen over 2 fits: **MIXED**

R@5 of zgs's p@swa minus zret's p@swa of the same split (zret's fits: its screen fit scr-zret and its full run's L-hotpotqa fit), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0030 WITHIN | +0.0009 WITHIN | -0.0100 LOSS zs | -0.0014 WITHIN | +0.0005 WITHIN | +0.0109 GAIN zs | MIXED |
| L-hotpotqa | +0.0078 GAIN | -0.0010 WITHIN | -0.0065 WITHIN | -0.0053 WITHIN zs | +0.0009 WITHIN | +0.0032 WITHIN zs | PROMISING |

Reads with a GAIN: 2 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
