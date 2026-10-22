U2d (docs/U2D_REFIT_ON_U2E.md), decision C (the pools and via together): zrckpv on the new build against zrckp on the base build.

# Screen over 2 fits: **MIXED**

R@5 of the screen's p@swa minus the decision's base fit's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0022 WITHIN | -0.0007 WITHIN | +0.0043 WITHIN zs | -0.0605 LOSS | -0.0198 LOSS | +0.0168 GAIN zs | MIXED |
| L-hotpotqa | -0.0031 WITHIN | +0.0004 WITHIN | +0.0040 WITHIN | -0.0451 LOSS zs | -0.0243 LOSS | +0.0150 GAIN zs | MIXED |

Reads with a GAIN: 2 of 12; with a LOSS: 4. PROMISING needs at least one GAIN and no LOSS.
