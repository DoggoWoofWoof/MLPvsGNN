A3z (docs/A3Z_WHOLE_POOL_Z.md): zrckpz (zrckp with step 1's whole-pool z-scores beside its z-scores) against zrckp.

# Screen over 2 fits: **NO_GAIN**

R@5 of zrckpz's p@swa minus zrckp's p@swa of the same split (zrckp's card fits scr-p1c3p-zrckp and scr-p1c3p-zrckp-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0018 WITHIN | -0.0006 WITHIN | -0.0015 WITHIN zs | -0.0018 WITHIN | -0.0042 WITHIN | -0.0153 LOSS zs | NO_GAIN |
| L-hotpotqa | -0.0014 WITHIN | -0.0003 WITHIN | -0.0023 WITHIN | +0.0029 WITHIN zs | -0.0005 WITHIN | +0.0039 WITHIN zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
