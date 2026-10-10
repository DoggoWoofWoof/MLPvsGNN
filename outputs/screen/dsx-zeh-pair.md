# Screen over 2 fits: **NO_GAIN**

R@5 of zeh's p@swa minus zrc's p@swa of the same split (zrc's screen fits scr-zrct and scr-zrct-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0016 WITHIN | -0.0016 WITHIN | -0.0107 LOSS zs | -0.0038 WITHIN | -0.0031 WITHIN | -0.0102 LOSS zs | NO_GAIN |
| L-hotpotqa | -0.0036 WITHIN | -0.0045 WITHIN | +0.0012 WITHIN | -0.0016 WITHIN zs | -0.0089 LOSS | -0.0115 LOSS zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 4. PROMISING needs at least one GAIN and no LOSS.
