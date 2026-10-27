A11x (docs/A11X_FOURTH_FAMILY_ZSPX.md): zspx4 (zspx whose three steps also read our LLM-free phrase links as a fourth family) against zspx.

# Screen over 2 fits: **NO_GAIN**

R@5 of zspx4's p@swa minus zspx's p@swa of the same split (zspx's card fits scr-g5-zspx and scr-g5-zspx-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0012 WITHIN | +0.0025 WITHIN | -0.0020 WITHIN zs | +0.0011 WITHIN | -0.0012 WITHIN | -0.0028 WITHIN zs | NO_GAIN |
| L-hotpotqa | +0.0011 WITHIN | +0.0002 WITHIN | -0.0020 WITHIN | -0.0017 WITHIN zs | +0.0015 WITHIN | -0.0145 LOSS zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
