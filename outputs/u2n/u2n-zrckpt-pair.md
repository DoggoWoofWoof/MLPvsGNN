U2n (docs/U2N_TYPED_CHANNELS.md): zrckpt against KG's zrckp on the T4, deciding.

# Screen over 2 fits: **MIXED**

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0002 WITHIN | +0.0000 WITHIN | +0.0098 GAIN zs | +0.0026 WITHIN | +0.0042 WITHIN | -0.0053 WITHIN zs | PROMISING |
| L-hotpotqa | +0.0001 WITHIN | +0.0011 WITHIN | +0.0019 WITHIN | +0.0080 GAIN zs | -0.0005 WITHIN | -0.0108 LOSS zs | MIXED |

Reads with a GAIN: 2 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
