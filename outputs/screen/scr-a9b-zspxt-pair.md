A9b (docs/A9B_BOUND_ALONE.md): zspxt (zspx with its three increments bounded together, no dropout) against zspx.

# Screen over 2 fits: **MIXED**

R@5 of zspxt's p@swa minus zspx's p@swa of the same split (zspx's card fits scr-g5-zspx and scr-g5-zspx-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0012 WITHIN | +0.0024 WITHIN | +0.0124 GAIN zs | -0.0004 WITHIN | -0.0023 WITHIN | -0.0054 WITHIN zs | PROMISING |
| L-hotpotqa | -0.0008 WITHIN | +0.0005 WITHIN | -0.0116 LOSS | -0.0034 WITHIN zs | -0.0004 WITHIN | -0.0237 LOSS zs | NO_GAIN |

Reads with a GAIN: 1 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
