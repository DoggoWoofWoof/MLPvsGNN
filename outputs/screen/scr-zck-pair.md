# Screen over 2 fits: **MIXED**

R@5 of zck's p@swa minus zrc's p@swa of the same split (zrc's screen fits scr-zrct and scr-zrct-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0000 WITHIN | +0.0011 WITHIN | +0.0039 WITHIN zs | +0.0000 WITHIN | -0.0041 WITHIN | +0.0160 GAIN zs | PROMISING |
| L-hotpotqa | -0.0001 WITHIN | +0.0015 WITHIN | +0.0030 WITHIN | -0.0032 WITHIN zs | -0.0082 LOSS | +0.0402 GAIN zs | MIXED |

Reads with a GAIN: 2 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
