A6ex (docs/A6EX_LINK_WORDS_ADOPTED.md): zspxl (zspx with A6e's link-weighted inputs in each of its three steps) against zspx.

# Screen over 2 fits: **NO_GAIN**

R@5 of zspxl's p@swa minus zspx's p@swa of the same split (zspx's card fits scr-g5-zspx and scr-g5-zspx-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0007 WITHIN | +0.0025 WITHIN | -0.0195 LOSS zs | +0.0035 WITHIN | -0.0055 WITHIN | -0.0064 WITHIN zs | NO_GAIN |
| L-hotpotqa | +0.0003 WITHIN | +0.0021 WITHIN | +0.0002 WITHIN | +0.0003 WITHIN zs | +0.0038 WITHIN | -0.0078 WITHIN zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
