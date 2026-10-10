# Screen over 2 fits: **NO_GAIN**

R@5 of zds's p@swa minus zrc's p@swa of the same split (zrc's screen fits scr-zrct and scr-zrct-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0259 LOSS | -0.0084 LOSS | -0.0180 LOSS zs | -0.0146 LOSS | -0.0175 LOSS | -0.0813 LOSS zs | NO_GAIN |
| L-hotpotqa | -0.0298 LOSS | -0.0089 LOSS | -0.0093 LOSS | -0.0084 LOSS zs | -0.0242 LOSS | -0.0542 LOSS zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 12. PROMISING needs at least one GAIN and no LOSS.
