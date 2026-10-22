U2d (docs/U2D_REFIT_ON_U2E.md), decision C (the pools and via together): zspkv on the new build against zspk on the base build.

# Screen over 2 fits: **MIXED**

R@5 of the screen's p@swa minus the decision's base fit's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0024 WITHIN | +0.0009 WITHIN | +0.0121 GAIN zs | -0.0625 LOSS | -0.0187 LOSS | -0.0012 WITHIN zs | MIXED |
| L-hotpotqa | -0.0053 WITHIN | +0.0026 WITHIN | -0.0072 WITHIN | -0.0427 LOSS zs | -0.0183 LOSS | +0.0094 WITHIN zs | NO_GAIN |

Reads with a GAIN: 1 of 12; with a LOSS: 4. PROMISING needs at least one GAIN and no LOSS.
