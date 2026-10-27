A5cp (docs/A5CP_P1GB_POOLS.md): zspkc (zspk with zck's coupled kind offsets on typed graphs (zret's output layer and base weight, trained jointly), before its propagation) against zspk, both on P1g-b's pools.

# Screen over 2 fits: **PROMISING**

R@5 of zspkc's p@swa minus zspk's p@swa of the same split (zspk's card fits scr-p1gb-zspk and scr-p1gb-zspk-hp, P1g-b's pools), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0023 WITHIN | +0.0003 WITHIN | +0.0132 GAIN zs | +0.0004 WITHIN | +0.0021 WITHIN | +0.0119 GAIN zs | PROMISING |
| L-hotpotqa | -0.0007 WITHIN | +0.0001 WITHIN | +0.0122 GAIN | +0.0045 WITHIN zs | +0.0044 WITHIN | +0.0455 GAIN zs | PROMISING |

Reads with a GAIN: 4 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
