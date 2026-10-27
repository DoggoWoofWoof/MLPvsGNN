A4he (docs/A4HE_SEMB_DROPOUT.md): zrckphe (zrckp trained with A4he's dropout (0.1 per question on SEMB and with it rmatch's chain term, the eight fixed blocks kept)) against zrckp.

# Screen over 2 fits: **MIXED**

R@5 of zrckphe's p@swa minus zrckp's p@swa of the same split (zrckp's card fits scr-p1c3p-zrckp and scr-p1c3p-zrckp-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0002 WITHIN | -0.0008 WITHIN | -0.0706 LOSS zs | -0.0036 WITHIN | -0.0072 WITHIN | +0.0181 GAIN zs | MIXED |
| L-hotpotqa | -0.0013 WITHIN | -0.0005 WITHIN | +0.0016 WITHIN | +0.0058 WITHIN zs | -0.0024 WITHIN | -0.0010 WITHIN zs | NO_GAIN |

Reads with a GAIN: 1 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
