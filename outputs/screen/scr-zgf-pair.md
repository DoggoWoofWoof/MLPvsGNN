# Screen over 2 fits: **NO_GAIN**

R@5 of zgf's p@swa minus zrm's p@swa of the same split (zrm's screen fits scr-zrm and scr-zrm-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0023 WITHIN | -0.0035 WITHIN | +0.0060 WITHIN zs | -0.0016 WITHIN | +0.0043 WITHIN | -0.0009 WITHIN zs | NO_GAIN |
| L-hotpotqa | +0.0015 WITHIN | -0.0143 LOSS | -0.0599 LOSS | -0.0437 LOSS zs | -0.0055 WITHIN | -0.0008 WITHIN zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 3. PROMISING needs at least one GAIN and no LOSS.
