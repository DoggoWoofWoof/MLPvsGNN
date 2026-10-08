# Screen over 2 fits: **MIXED**

R@5 of relz's p@swa minus rel's p@swa of the same split (rel's screen fits scr-rel and scr-rel-hp), on the six s1eval carves (95% question bootstrap; docs/SCREENS.md's rule over every read). zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp | fit's own verdict |
|---|---|---|---|---|---|---|---|
| L-musique | -0.0196 LOSS | +0.0051 WITHIN | +0.1563 GAIN zs | +0.0018 WITHIN | +0.0005 WITHIN | +0.0127 WITHIN zs | MIXED |
| L-hotpotqa | -0.0242 LOSS | +0.0030 WITHIN | +0.0051 WITHIN | -0.0117 LOSS zs | -0.0018 WITHIN | +0.0601 GAIN zs | MIXED |

Reads with a GAIN: 2 of 12; with a LOSS: 3. PROMISING needs at least one GAIN and no LOSS.
