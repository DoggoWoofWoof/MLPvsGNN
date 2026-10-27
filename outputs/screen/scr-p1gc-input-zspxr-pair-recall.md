P1g-c (docs/P1GC_ADOPTED_ON_P1GB.md), the input call: zspxr (zspx with R1's head before its first step) on P1g-b's pools against zspx on P1g-b's pools.

# scr-p1gc-input-zspxr-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of zspxr's p@swa minus zspx's p@swa of the same split (zspx's card fits on P1g-b's pools), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/P1GC_ADOPTED_ON_P1GB.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zspx R@5 (P1g-b's) | zspxr R@5 (P1g-b's) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7992 | 0.8027 | +0.0035 [+0.0025, +0.0045] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9131 | 0.9119 | -0.0012 [-0.0028, +0.0005] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5206 | 0.5127 | -0.0079 [-0.0134, -0.0021] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9105 | 0.9074 | -0.0031 [-0.0055, -0.0009] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9289 | 0.9307 | +0.0018 [+0.0002, +0.0034] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3513 | 0.3338 | -0.0175 [-0.0272, -0.0085] | -0.0070, -0.0018 | 0.0102 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | 0.7996 | 0.8006 | +0.0010 [+0.0000, +0.0020] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9120 | 0.9137 | +0.0017 [+0.0000, +0.0033] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5814 | 0.5839 | +0.0025 [-0.0026, +0.0076] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8575 | 0.8519 | -0.0055 [-0.0084, -0.0028] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9338 | 0.9346 | +0.0007 [-0.0009, +0.0023] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3072 | 0.2715 | -0.0357 [-0.0453, -0.0259] | +0.0102, +0.0129 | 0.0233 | LOSS | LOSS |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
