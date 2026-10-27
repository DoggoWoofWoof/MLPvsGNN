G6b (docs/G6B_GATED_BACKOFF.md): zg6b against KG's zspk on the T4, deciding.

# Screen over 2 fits: **MIXED**

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0114 GAIN | +0.0011 WITHIN | -0.0094 LOSS zs | +0.0042 WITHIN | +0.0060 WITHIN | -0.0367 LOSS zs | MIXED |
| L-hotpotqa | +0.0078 GAIN | -0.0009 WITHIN | +0.0180 GAIN | -0.0162 LOSS zs | +0.0177 GAIN | -0.0243 LOSS zs | MIXED |

Reads with a GAIN: 4 of 12; with a LOSS: 4. PROMISING needs at least one GAIN and no LOSS.
