A2xmx (docs/A2XMX_ZTOP50_ADOPTED.md): zspxxm (zspx with zret's reference cut to the pool's top 50 retrieved rows by rrf (ztop50's form), before its three increments) against zspx.

# Screen over 2 fits: **NO_GAIN**

R@5 of zspxxm's p@swa minus zspx's p@swa of the same split (zspx's card fits scr-g5-zspx and scr-g5-zspx-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0026 WITHIN | +0.0024 WITHIN | -0.0124 LOSS zs | +0.0006 WITHIN | -0.0008 WITHIN | +0.0088 WITHIN zs | NO_GAIN |
| L-hotpotqa | -0.0020 WITHIN | -0.0003 WITHIN | -0.0038 WITHIN | -0.0032 WITHIN zs | -0.0030 WITHIN | -0.0254 LOSS zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
