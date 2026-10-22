U2d (docs/U2D_REFIT_ON_U2E.md), decision G (the pools, for zg6): zg6 on the new build against zg6 on the base build.

# Screen over 2 fits: **MIXED**

R@5 of the screen's p@swa minus the decision's base fit's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0011 WITHIN | +0.0035 WITHIN | +0.0220 GAIN zs | -0.0514 LOSS | -0.0110 LOSS | -0.0808 LOSS zs | MIXED |
| L-hotpotqa | +0.0022 WITHIN | +0.0019 WITHIN | +0.0048 WITHIN | -0.0499 LOSS zs | -0.0124 LOSS | -0.0408 LOSS zs | NO_GAIN |

Reads with a GAIN: 1 of 12; with a LOSS: 6. PROMISING needs at least one GAIN and no LOSS.
