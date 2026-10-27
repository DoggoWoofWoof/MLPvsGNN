A8ex (docs/A8EX_QDEPTH_PRIOR_ADOPTED.md): zrckprqe (zrckpr plus round 11's depth prior (A8e's), added to its final score) against zrckpr.

# Screen over 2 fits: **PROMISING**

R@5 of zrckprqe's p@swa minus zrckpr's p@swa of the same split (zrckpr's card fits scr-r1-zrckpr and scr-r1-zrckpr-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0014 WITHIN | -0.0017 WITHIN | -0.0002 WITHIN zs | -0.0003 WITHIN | -0.0029 WITHIN | +0.0066 WITHIN zs | NO_GAIN |
| L-hotpotqa | -0.0002 WITHIN | -0.0005 WITHIN | -0.0067 WITHIN | -0.0008 WITHIN zs | -0.0026 WITHIN | +0.0315 GAIN zs | PROMISING |

Reads with a GAIN: 1 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
