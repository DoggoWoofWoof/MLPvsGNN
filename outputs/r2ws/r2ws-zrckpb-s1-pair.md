R2WS (docs/R2WS_R2_SEEDS_UNDER_WQ.md): zrckpb at seed 1 against zrckp at seed 1 under WQ on the T4, with the rule joining the seeds.

# Screen over 2 fits: **MIXED**

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| WQ-musique | +0.0266 GAIN zs | -0.0004 WITHIN | +0.0373 GAIN zs | +0.0015 WITHIN | +0.0036 WITHIN | +0.0134 GAIN | PROMISING |
| WQ-hotpotqa | +0.0087 GAIN zs | -0.0026 WITHIN | -0.0175 LOSS | -0.0125 LOSS zs | +0.0022 WITHIN | +0.0024 WITHIN | MIXED |

Reads with a GAIN: 4 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
