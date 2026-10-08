# Full run of zrs against zret: **ADOPT**

R@5 of zrs's p@swa minus zret's p@swa of the same split, on the six s1eval carves (95% question-bootstrap interval; call by docs/SCREENS.md's rule). Bold: step 1's eleven primary reads. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.1205 GAIN** | **+0.0000 WITHIN** | **+0.0000 WITHIN** | **+0.0000 WITHIN** | **+0.0000 WITHIN** | **+0.0912 GAIN zs** |
| L-metaqa | **+0.0000 WITHIN zs** | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN zs |
| L-squad | +0.1159 GAIN | **+0.0000 WITHIN zs** | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0795 GAIN zs |
| L-musique | +0.1140 GAIN | +0.0000 WITHIN | **+0.0000 WITHIN zs** | +0.0000 WITHIN | +0.0000 WITHIN | +0.0857 GAIN zs |
| L-hotpotqa | +0.1223 GAIN | +0.0000 WITHIN | +0.0000 WITHIN | **+0.0000 WITHIN zs** | +0.0000 WITHIN | +0.0905 GAIN zs |
| L-2wiki | +0.1178 GAIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | **+0.0000 WITHIN zs** | +0.0770 GAIN zs |

Primary reads with a GAIN: 2 of 11. Reads with a LOSS: 0 of 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/FULL_ROUND15.md).
