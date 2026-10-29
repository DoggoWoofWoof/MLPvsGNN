R2EG (docs/R2EG_R2E_GNN_UNDER_WQ.md): zg6e against zg6 under WQ on the T4, deciding.

# Screen over 2 fits: **MIXED**

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| WQ-musique | -0.0443 LOSS zs | +0.0004 WITHIN | -0.0180 LOSS zs | +0.0036 WITHIN | -0.0010 WITHIN | +0.0015 WITHIN | NO_GAIN |
| WQ-hotpotqa | -0.0320 LOSS zs | -0.0006 WITHIN | -0.0033 WITHIN | +0.0097 GAIN zs | -0.0015 WITHIN | -0.0078 WITHIN | MIXED |

Reads with a GAIN: 1 of 12; with a LOSS: 3. PROMISING needs at least one GAIN and no LOSS.
