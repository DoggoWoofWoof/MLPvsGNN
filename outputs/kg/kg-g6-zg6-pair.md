# Screen over 2 fits: **MIXED**

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0114 GAIN | +0.0018 WITHIN | -0.0009 WITHIN zs | +0.0041 WITHIN | +0.0084 GAIN | -0.0753 LOSS zs | MIXED |
| L-hotpotqa | +0.0078 GAIN | -0.0017 WITHIN | +0.0199 GAIN | -0.0159 LOSS zs | +0.0198 GAIN | -0.0210 LOSS zs | MIXED |

Reads with a GAIN: 5 of 12; with a LOSS: 3. PROMISING needs at least one GAIN and no LOSS.
