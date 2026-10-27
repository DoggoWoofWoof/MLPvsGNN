A13x (docs/A13X_BRIDGE_HEAD_ZSPX.md): zspxb (zspx with zbr's bridge head added to its score before its first step) against zspx.

# Screen over 2 fits: **NO_GAIN**

R@5 of zspxb's p@swa minus zspx's p@swa of the same split (zspx's card fits scr-g5-zspx and scr-g5-zspx-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0002 WITHIN | +0.0004 WITHIN | -0.0119 LOSS zs | +0.0014 WITHIN | -0.0011 WITHIN | +0.0065 WITHIN zs | NO_GAIN |
| L-hotpotqa | -0.0003 WITHIN | +0.0023 WITHIN | -0.0065 WITHIN | +0.0013 WITHIN zs | -0.0015 WITHIN | -0.0133 LOSS zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
