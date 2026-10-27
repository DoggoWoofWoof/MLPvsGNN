A3z (docs/A3Z_WHOLE_POOL_Z.md): zspkz (zspk with step 1's whole-pool z-scores beside its z-scores, before its propagation) against zspk.

# Screen over 2 fits: **MIXED**

R@5 of zspkz's p@swa minus zspk's p@swa of the same split (zspk's card fits scr-p1c3-zspk and scr-p1c3-zspk-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0014 WITHIN | +0.0000 WITHIN | +0.0183 GAIN zs | -0.0033 WITHIN | -0.0089 LOSS | -0.0122 LOSS zs | MIXED |
| L-hotpotqa | -0.0025 WITHIN | -0.0003 WITHIN | -0.0040 WITHIN | -0.0072 WITHIN zs | -0.0003 WITHIN | +0.0168 GAIN zs | PROMISING |

Reads with a GAIN: 2 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
