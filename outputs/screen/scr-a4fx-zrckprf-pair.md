A4fx (docs/A4FX_FAMILY_DROPOUT_ADOPTED.md): zrckprf (zrckpr trained with A4f's family dropout and the R1 cell (0.2 per question on each block, the KT term, the chain term and R1's term)) against zrckpr.

# Screen over 2 fits: **NO_GAIN**

R@5 of zrckprf's p@swa minus zrckpr's p@swa of the same split (zrckpr's card fits scr-r1-zrckpr and scr-r1-zrckpr-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0018 WITHIN | -0.0030 WITHIN | -0.1663 LOSS zs | -0.0045 WITHIN | -0.0051 WITHIN | -0.0220 LOSS zs | NO_GAIN |
| L-hotpotqa | -0.0073 WITHIN | -0.0005 WITHIN | -0.0077 LOSS | +0.0054 WITHIN zs | -0.0050 WITHIN | +0.0120 WITHIN zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 3. PROMISING needs at least one GAIN and no LOSS.
