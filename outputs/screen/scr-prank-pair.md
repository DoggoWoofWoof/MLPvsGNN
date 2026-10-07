# Screen over 2 fits: **MIXED**

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0310 LOSS | +0.0011 WITHIN | +0.2016 GAIN zs | -0.0080 LOSS | -0.0068 WITHIN | +0.0269 GAIN zs | MIXED |
| L-hotpotqa | -0.0320 LOSS | -0.0035 WITHIN | -0.0067 WITHIN | -0.0255 LOSS zs | -0.0004 WITHIN | +0.0054 WITHIN zs | NO_GAIN |

Reads with a GAIN: 2 of 12; with a LOSS: 4. PROMISING needs at least one GAIN and no LOSS.
