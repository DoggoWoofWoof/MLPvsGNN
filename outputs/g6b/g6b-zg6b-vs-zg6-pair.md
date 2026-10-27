G6b (docs/G6B_GATED_BACKOFF.md): zg6b against KG's zg6 on the T4, beside, deciding nothing.

# Screen over 2 fits: **MIXED**

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0000 WITHIN | -0.0007 WITHIN | -0.0085 LOSS zs | +0.0001 WITHIN | -0.0024 WITHIN | +0.0386 GAIN zs | MIXED |
| L-hotpotqa | -0.0000 WITHIN | +0.0008 WITHIN | -0.0019 WITHIN | -0.0003 WITHIN zs | -0.0020 WITHIN | -0.0033 WITHIN zs | NO_GAIN |

Reads with a GAIN: 1 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
