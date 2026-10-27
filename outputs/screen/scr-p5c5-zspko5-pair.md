P5c5 (docs/P5C5_FIVE_FOLDS.md): zspko5 (zspk plus the head over the four five-fold columns and P5b's log p, before its propagation) against zspk.

# Screen over 2 fits: **MIXED**

R@5 of zspko5's p@swa minus zspk's p@swa of the same split (zspk's card fits scr-p1c3-zspk and scr-p1c3-zspk-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0043 WITHIN | -0.0008 WITHIN | +0.0146 GAIN zs | +0.0007 WITHIN | -0.0014 WITHIN | -0.0142 LOSS zs | MIXED |
| L-hotpotqa | +0.0020 WITHIN | +0.0008 WITHIN | +0.0053 WITHIN | +0.0009 WITHIN zs | +0.0003 WITHIN | +0.0046 WITHIN zs | NO_GAIN |

Reads with a GAIN: 1 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
