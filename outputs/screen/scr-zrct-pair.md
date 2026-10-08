# Screen over 2 fits: **PROMISING**

R@5 of zrc's p@swa minus zrm's p@swa of the same split (zrm's screen fits scr-zrm and scr-zrm-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0007 WITHIN | -0.0006 WITHIN | +0.0004 WITHIN zs | +0.0001 WITHIN | +0.0024 WITHIN | +0.0257 GAIN zs | PROMISING |
| L-hotpotqa | -0.0008 WITHIN | -0.0003 WITHIN | +0.0045 WITHIN | -0.0006 WITHIN zs | +0.0051 WITHIN | +0.0450 GAIN zs | PROMISING |

Reads with a GAIN: 2 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
