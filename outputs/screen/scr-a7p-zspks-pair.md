A7p (docs/A7P_P1GB_POOLS.md): zspks (zspk plus the A7 head before its propagation) against zspk, both on P1g-b's pools.

# Screen over 2 fits: **PROMISING**

R@5 of zspks's p@swa minus zspk's p@swa of the same split (zspk's card fits scr-p1gb-zspk and scr-p1gb-zspk-hp, P1g-b's pools), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0010 WITHIN | +0.0013 WITHIN | +0.0012 WITHIN zs | +0.0017 WITHIN | +0.0044 WITHIN | +0.0036 WITHIN zs | NO_GAIN |
| L-hotpotqa | +0.0005 WITHIN | +0.0015 WITHIN | +0.0084 GAIN | +0.0049 WITHIN zs | +0.0030 WITHIN | +0.0042 WITHIN zs | PROMISING |

Reads with a GAIN: 1 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
