U1c (docs/U1C_RETRAIN_ON_U.md): zsp on U1d's graph and step 4h's pools against zsp on today's.

# Screen over 2 fits: **NO_GAIN**

R@5 of zsp's p@swa minus zsp's p@swa of the same split (zsp's card fits scr-zspg and scr-zspg-hp, today's graph and pools), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0041 WITHIN | +0.0008 WITHIN | -0.0189 LOSS zs | -0.0634 LOSS | -0.0112 LOSS | -0.0562 LOSS zs | NO_GAIN |
| L-hotpotqa | +0.0056 WITHIN | +0.0012 WITHIN | -0.0034 WITHIN | -0.0397 LOSS zs | -0.0139 LOSS | +0.0102 WITHIN zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 6. PROMISING needs at least one GAIN and no LOSS.
