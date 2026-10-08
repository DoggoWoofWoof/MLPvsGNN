# Screen over 2 fits: **NO_GAIN**

R@5 of zrk's p@swa minus zrm's p@swa of the same split (zrm's screen fits scr-zrm and scr-zrm-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0003 WITHIN | +0.0002 WITHIN | +0.0007 WITHIN zs | +0.0005 WITHIN | +0.0015 WITHIN | +0.0069 WITHIN zs | NO_GAIN |
| L-hotpotqa | +0.0006 WITHIN | +0.0014 WITHIN | +0.0003 WITHIN | -0.0041 WITHIN zs | +0.0004 WITHIN | +0.0016 WITHIN zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
