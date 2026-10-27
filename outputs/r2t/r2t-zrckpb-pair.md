R2t (docs/R2T_BRIDGE_EM_ON_T4.md): zrckpb (zrckp + R2's bridge head) against KG's zrckp on the T4, deciding.

# Screen over 2 fits: **MIXED**

R@5 of zrckpb's p@swa minus KG's seed-0 zrckp's p@swa of the same split, both on U2d's base build on the T4, on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0054 WITHIN | -0.0010 WITHIN | +0.0091 GAIN zs | +0.0022 WITHIN | +0.0026 WITHIN | +0.0120 GAIN zs | PROMISING |
| L-hotpotqa | +0.0018 WITHIN | -0.0033 WITHIN | -0.0133 LOSS | -0.0050 WITHIN zs | -0.0014 WITHIN | +0.0098 WITHIN zs | NO_GAIN |

Reads with a GAIN: 2 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
