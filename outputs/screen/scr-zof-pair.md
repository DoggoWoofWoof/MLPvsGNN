# Screen over 2 fits: **NO_GAIN**

R@5 of zof's p@swa minus zrc's p@swa of the same split (zrc's screen fits scr-zrct and scr-zrct-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0043 WITHIN | -0.0169 LOSS | -0.0377 LOSS zs | -0.0084 LOSS | -0.0019 WITHIN | -0.0033 WITHIN zs | NO_GAIN |
| L-hotpotqa | +0.0046 WITHIN | -0.0168 LOSS | -0.0873 LOSS | -0.0494 LOSS zs | -0.0117 LOSS | -0.0247 LOSS zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 8. PROMISING needs at least one GAIN and no LOSS.
