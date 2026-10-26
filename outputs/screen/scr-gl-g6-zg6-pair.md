GL (docs/GL_GPU_LINE_ON_L40S.md), G6's call on U2d's L40S fits: zg6 (the free GNN) against zspk on U2d's base build.

# Screen over 2 fits: **MIXED**

R@5 of zg6's p@swa minus zspk's p@swa of the same split, both U2d's base-build fits on the L40S, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0102 GAIN | -0.0012 WITHIN | +0.0063 WITHIN zs | +0.0030 WITHIN | +0.0070 WITHIN | -0.0795 LOSS zs | MIXED |
| L-hotpotqa | +0.0068 WITHIN | +0.0001 WITHIN | +0.0138 GAIN | -0.0203 LOSS zs | +0.0101 GAIN | -0.0584 LOSS zs | MIXED |

Reads with a GAIN: 3 of 12; with a LOSS: 3. PROMISING needs at least one GAIN and no LOSS.
