# Full run of zrckpw (KT_W beside KT) against zrckp: NOT_ADOPTED

R@5 of zrckpw's p@swa minus zrckp's seed-0 p@swa of the same split on U2d's base build, on the six s1eval carves (95% question bootstrap). Each read is called under its floor from zrckp's own seed null on that split (docs/U2H_FULL_RUN.md, section 4). Bold: the eleven primary reads. zs: zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **-0.0048 WITHIN (floor 0.0075)** | **-0.0012 WITHIN (floor 0.0075)** | **+0.0006 WITHIN (floor 0.0077)** | **-0.0028 WITHIN (floor 0.0075)** | **+0.0007 WITHIN (floor 0.0075)** | **-0.0131 WITHIN zs (floor 0.0161)** |
| L-metaqa | **-0.0028 WITHIN zs (floor 0.0139)** | +0.0026 WITHIN (floor 0.0075) | -0.0002 WITHIN (floor 0.0087) | +0.0000 WITHIN (floor 0.0075) | +0.0059 WITHIN (floor 0.0094) | +0.0087 GAIN zs (floor 0.0083) |
| L-squad | -0.0024 WITHIN (floor 0.0075) | **+0.0018 WITHIN zs (floor 0.0075)** | -0.0012 WITHIN (floor 0.0075) | +0.0000 WITHIN (floor 0.0075) | +0.0003 WITHIN (floor 0.0075) | -0.0021 WITHIN zs (floor 0.0247) |
| L-musique | -0.0001 WITHIN (floor 0.0075) | -0.0015 WITHIN (floor 0.0075) | **-0.0021 WITHIN zs (floor 0.0136)** | -0.0002 WITHIN (floor 0.0075) | +0.0039 WITHIN (floor 0.0075) | -0.0003 WITHIN zs (floor 0.0182) |
| L-hotpotqa | +0.0011 WITHIN (floor 0.0075) | -0.0019 WITHIN (floor 0.0075) | +0.0039 WITHIN (floor 0.0120) | **-0.0069 WITHIN zs (floor 0.0083)** | +0.0078 GAIN (floor 0.0075) | -0.0057 WITHIN zs (floor 0.0322) |
| L-2wiki | +0.0005 WITHIN (floor 0.0075) | +0.0002 WITHIN (floor 0.0075) | +0.0009 WITHIN (floor 0.0214) | -0.0008 WITHIN (floor 0.0135) | **-0.0024 WITHIN zs (floor 0.0179)** | +0.0035 WITHIN zs (floor 0.0198) |

Primary GAINs: none. LOSSes: none. GAINs in all: 2 of 36.
