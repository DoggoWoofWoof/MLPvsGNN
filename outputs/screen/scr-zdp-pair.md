# Screen over 2 fits: **MIXED**

R@5 of zdp's p@swa minus zsp's p@swa of the same split (zsp's CPU screen fits scr-zsp and scr-zsp-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0009 WITHIN | +0.0035 WITHIN | -0.0029 WITHIN zs | -0.0010 WITHIN | -0.0001 WITHIN | -0.0133 LOSS zs | NO_GAIN |
| L-hotpotqa | +0.0001 WITHIN | -0.0013 WITHIN | -0.0071 WITHIN | -0.0016 WITHIN zs | -0.0025 WITHIN | +0.0173 GAIN zs | PROMISING |

Reads with a GAIN: 1 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
