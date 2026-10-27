A2xmx (docs/A2XMX_ZTOP50_ADOPTED.md): zrckprxm (zrckpr with zret's reference cut to the pool's top 50 retrieved rows by rrf (ztop50's form)) against zrckpr.

# Screen over 2 fits: **NO_GAIN**

R@5 of zrckprxm's p@swa minus zrckpr's p@swa of the same split (zrckpr's card fits scr-r1-zrckpr and scr-r1-zrckpr-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0008 WITHIN | -0.0016 WITHIN | -0.0032 WITHIN zs | -0.0023 WITHIN | -0.0005 WITHIN | +0.0094 WITHIN zs | NO_GAIN |
| L-hotpotqa | -0.0064 WITHIN | +0.0000 WITHIN | -0.0057 WITHIN | -0.0018 WITHIN zs | -0.0088 LOSS | -0.0029 WITHIN zs | NO_GAIN |

Reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
