# Screen over 2 fits: **MIXED**

R@5 of zsep's p@swa minus zret's p@swa of the same split (zret's fits: its screen fit scr-zret and its full run's L-hotpotqa fit), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0043 WITHIN | -0.0012 WITHIN | -0.0687 LOSS zs | -0.0030 WITHIN | -0.0007 WITHIN | +0.0003 WITHIN zs | NO_GAIN |
| L-hotpotqa | +0.0104 GAIN | -0.0015 WITHIN | -0.0101 LOSS | -0.0026 WITHIN zs | -0.0003 WITHIN | -0.0006 WITHIN zs | MIXED |

Reads with a GAIN: 1 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
