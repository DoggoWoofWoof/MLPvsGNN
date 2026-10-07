# Screen over 2 fits: **MIXED**

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0060 WITHIN | -0.0018 WITHIN | +0.0436 GAIN zs | -0.0022 WITHIN | -0.0018 WITHIN | -0.0220 LOSS zs | MIXED |
| L-hotpotqa | -0.0084 LOSS | -0.0009 WITHIN | +0.0032 WITHIN | -0.0132 LOSS zs | +0.0050 WITHIN | -0.0005 WITHIN zs | NO_GAIN |

Reads with a GAIN: 1 of 12; with a LOSS: 3. PROMISING needs at least one GAIN and no LOSS.
