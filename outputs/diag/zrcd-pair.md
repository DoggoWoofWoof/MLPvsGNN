# Screen over 2 fits: **PROMISING**

zrm's screen fits read with zrc's chain entries: trained on rmatch's entries, read with zrc's. A diagnosis (docs/SCREENS.md, 'zrcd'); it decides nothing about any arm.

R@5 of zrcd's p@swa minus zrm's p@swa of the same split (zrm's screen fits scr-zrm and scr-zrm-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN zs | +0.0000 WITHIN | +0.0000 WITHIN | +0.0215 GAIN zs | PROMISING |
| L-hotpotqa | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN zs | +0.0000 WITHIN | +0.0444 GAIN zs | PROMISING |

Reads with a GAIN: 2 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
