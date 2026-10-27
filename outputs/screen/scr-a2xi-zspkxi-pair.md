A2xi (docs/A2XI_PRANK_REPLACE.md): zspkxi (zspk with each fixed block's z-score replaced by prank's hi and lo, before its propagation) against zspk.

# Screen over 2 fits: **MIXED**

R@5 of zspkxi's p@swa minus zspk's p@swa of the same split (zspk's card fits scr-p1c3-zspk and scr-p1c3-zspk-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | +0.0027 WITHIN | -0.0005 WITHIN | +0.0110 GAIN zs | -0.0054 WITHIN | -0.0044 WITHIN | -0.0236 LOSS zs | MIXED |
| L-hotpotqa | -0.0103 LOSS | -0.0029 WITHIN | -0.0051 WITHIN | -0.0132 LOSS zs | +0.0015 WITHIN | +0.0068 WITHIN zs | NO_GAIN |

Reads with a GAIN: 1 of 12; with a LOSS: 3. PROMISING needs at least one GAIN and no LOSS.
