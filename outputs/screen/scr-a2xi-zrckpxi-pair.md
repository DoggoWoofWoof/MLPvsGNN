A2xi (docs/A2XI_PRANK_REPLACE.md): zrckpxi (zrckp with each fixed block's z-score replaced by prank's hi and lo) against zrckp.

# Screen over 2 fits: **MIXED**

R@5 of zrckpxi's p@swa minus zrckp's p@swa of the same split (zrckp's card fits scr-p1c3p-zrckp and scr-p1c3p-zrckp-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0025 WITHIN | -0.0014 WITHIN | -0.0026 WITHIN zs | -0.0049 WITHIN | +0.0034 WITHIN | +0.0231 GAIN zs | PROMISING |
| L-hotpotqa | +0.0017 WITHIN | -0.0031 WITHIN | -0.0181 LOSS | -0.0149 LOSS zs | -0.0011 WITHIN | +0.0032 WITHIN zs | NO_GAIN |

Reads with a GAIN: 1 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
