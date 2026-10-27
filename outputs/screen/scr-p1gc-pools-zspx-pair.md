P1g-c (docs/P1GC_ADOPTED_ON_P1GB.md), the pools call: zspx (G5's GNN as filed) on P1g-b's pools against zspx on P1c2's pools.

# Screen over 2 fits: **MIXED**

R@5 of zspx's p@swa minus zspx's p@swa of the same split (zspx's card fits scr-g5-zspx and scr-g5-zspx-hp, P1c2's pools), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0038 WITHIN | +0.0039 WITHIN | +0.0133 GAIN zs | +0.0031 WITHIN | -0.0062 WITHIN | +0.0309 GAIN zs | PROMISING |
| L-hotpotqa | -0.0043 WITHIN | -0.0009 WITHIN | -0.0087 LOSS | +0.0028 WITHIN zs | +0.0026 WITHIN | +0.0158 GAIN zs | MIXED |

Reads with a GAIN: 3 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
