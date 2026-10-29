R2WS (docs/R2WS_R2_SEEDS_UNDER_WQ.md): zrckpb at seed 2 against zrckp at seed 2 under WQ on the T4, with the rule joining the seeds.

# Screen over 2 fits: **MIXED**

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| WQ-musique | +0.0690 GAIN zs | -0.0011 WITHIN | +0.0241 GAIN zs | +0.0001 WITHIN | +0.0035 WITHIN | +0.0066 WITHIN | PROMISING |
| WQ-hotpotqa | +0.0426 GAIN zs | -0.0027 WITHIN | -0.0215 LOSS | -0.0115 LOSS zs | -0.0011 WITHIN | +0.0122 GAIN | MIXED |

Reads with a GAIN: 4 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
