U2d (docs/U2D_REFIT_ON_U2E.md), decision B (via, on the new pools): zrckpv on the new build against zrckp on the new build.

# Screen over 2 fits: **NO_GAIN**

R@5 of the screen's p@swa minus the decision's base fit's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0013 WITHIN | -0.0015 WITHIN | +0.0007 WITHIN zs | -0.0034 WITHIN | -0.0057 WITHIN | -0.0222 LOSS zs | NO_GAIN |
| L-hotpotqa | -0.0007 WITHIN | -0.0003 WITHIN | -0.0003 WITHIN | +0.0023 WITHIN zs | +0.0030 WITHIN | -0.0350 LOSS zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
