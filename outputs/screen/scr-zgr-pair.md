# Screen over 2 fits: **NO_GAIN**

R@5 of zgr's p@swa minus zrm's p@swa of the same split (zrm's screen fits scr-zrm and scr-zrm-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0034 WITHIN | -0.0118 LOSS | -0.0057 WITHIN zs | -0.0063 WITHIN | -0.0003 WITHIN | -0.0004 WITHIN zs | NO_GAIN |
| L-hotpotqa | +0.0026 WITHIN | -0.0141 LOSS | -0.0474 LOSS | -0.0263 LOSS zs | -0.0025 WITHIN | -0.0065 WITHIN zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 4. PROMISING needs at least one GAIN and no LOSS.
