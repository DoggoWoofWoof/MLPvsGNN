# Screen over 2 fits: **NO_GAIN**

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0041 WITHIN | -0.0002 WITHIN | -0.0288 LOSS zs | +0.0009 WITHIN | -0.0009 WITHIN | -0.0131 LOSS zs | NO_GAIN |
| L-hotpotqa | -0.0073 WITHIN | +0.0003 WITHIN | +0.0035 WITHIN | -0.0155 LOSS zs | -0.0026 WITHIN | -0.0053 WITHIN zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 3. PROMISING needs at least one GAIN and no LOSS.
