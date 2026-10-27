G6b2 (docs/G6B2_ROW_BACKOFF.md): zg6b2 against KG's zspk on the T4, deciding.

# Screen over 2 fits: **MIXED**

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0099 GAIN | +0.0017 WITHIN | -0.0366 LOSS zs | +0.0027 WITHIN | +0.0067 WITHIN | -0.0929 LOSS zs | MIXED |
| L-hotpotqa | +0.0079 GAIN | -0.0012 WITHIN | +0.0248 GAIN | -0.0182 LOSS zs | +0.0154 GAIN | -0.0139 WITHIN zs | MIXED |

Reads with a GAIN: 4 of 12; with a LOSS: 3. PROMISING needs at least one GAIN and no LOSS.
