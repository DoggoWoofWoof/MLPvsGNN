A4fx (docs/A4FX_FAMILY_DROPOUT_ADOPTED.md): zspxf (zspx trained with A4f's family dropout (0.2 per question on each block, the KT term and the chain term), before its three increments) against zspx.

# Screen over 2 fits: **NO_GAIN**

R@5 of zspxf's p@swa minus zspx's p@swa of the same split (zspx's card fits scr-g5-zspx and scr-g5-zspx-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0034 WITHIN | -0.0011 WITHIN | -0.1938 LOSS zs | -0.0006 WITHIN | -0.0041 WITHIN | -0.0057 WITHIN zs | NO_GAIN |
| L-hotpotqa | -0.0025 WITHIN | -0.0010 WITHIN | -0.0049 WITHIN | +0.0028 WITHIN zs | -0.0043 WITHIN | +0.0087 WITHIN zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
