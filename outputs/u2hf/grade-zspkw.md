# Full run of zspkw (KT_W beside KT) against zspk: NOT_ADOPTED

R@5 of zspkw's p@swa minus zspk's seed-0 p@swa of the same split on U2d's base build, on the six s1eval carves (95% question bootstrap). Each read is called under its floor from zspk's own seed null on that split (docs/U2H_FULL_RUN.md, section 4). Bold: the eleven primary reads. zs: zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0007 WITHIN (floor 0.0075)** | **-0.0008 WITHIN (floor 0.0075)** | **+0.0059 WITHIN (floor 0.0112)** | **+0.0022 WITHIN (floor 0.0075)** | **+0.0020 WITHIN (floor 0.0075)** | **-0.0096 WITHIN zs (floor 0.0144)** |
| L-metaqa | **+0.0175 GAIN zs (floor 0.0166)** | -0.0013 WITHIN (floor 0.0075) | +0.0038 WITHIN (floor 0.0075) | +0.0000 WITHIN (floor 0.0075) | -0.0012 WITHIN (floor 0.0075) | +0.0065 WITHIN zs (floor 0.0075) |
| L-squad | +0.0004 WITHIN (floor 0.0075) | **+0.0019 WITHIN zs (floor 0.0075)** | -0.0120 LOSS (floor 0.0075) | +0.0016 WITHIN (floor 0.0075) | -0.0088 LOSS (floor 0.0075) | +0.0061 WITHIN zs (floor 0.0075) |
| L-musique | +0.0011 WITHIN (floor 0.0075) | +0.0004 WITHIN (floor 0.0075) | **+0.0059 WITHIN zs (floor 0.0089)** | -0.0015 WITHIN (floor 0.0075) | +0.0023 WITHIN (floor 0.0075) | +0.0030 WITHIN zs (floor 0.0105) |
| L-hotpotqa | +0.0009 WITHIN (floor 0.0075) | +0.0008 WITHIN (floor 0.0075) | -0.0018 WITHIN (floor 0.0107) | **-0.0059 WITHIN zs (floor 0.0145)** | +0.0083 WITHIN (floor 0.0139) | -0.0030 WITHIN zs (floor 0.0405) |
| L-2wiki | +0.0013 WITHIN (floor 0.0075) | +0.0001 WITHIN (floor 0.0075) | +0.0003 WITHIN (floor 0.0159) | +0.0025 WITHIN (floor 0.0120) | **+0.0002 WITHIN zs (floor 0.0075)** | +0.0130 WITHIN zs (floor 0.0530) |

Primary GAINs: L-metaqa metaqa. LOSSes: L-squad musique, L-squad 2wiki. GAINs in all: 1 of 36.
