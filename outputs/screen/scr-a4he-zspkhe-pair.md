A4he (docs/A4HE_SEMB_DROPOUT.md): zspkhe (zspk trained with A4he's dropout (0.1 per question on SEMB and with it rmatch's chain term, the eight fixed blocks kept), before its propagation) against zspk.

# Screen over 2 fits: **MIXED**

R@5 of zspkhe's p@swa minus zspk's p@swa of the same split (zspk's card fits scr-p1c3-zspk and scr-p1c3-zspk-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0013 WITHIN | -0.0021 WITHIN | -0.1036 LOSS zs | -0.0008 WITHIN | -0.0016 WITHIN | -0.0039 WITHIN zs | NO_GAIN |
| L-hotpotqa | -0.0001 WITHIN | -0.0003 WITHIN | +0.0006 WITHIN | +0.0099 GAIN zs | -0.0025 WITHIN | +0.0312 GAIN zs | PROMISING |

Reads with a GAIN: 2 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
