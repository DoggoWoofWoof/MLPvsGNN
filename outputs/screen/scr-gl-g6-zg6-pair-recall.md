GL (docs/GL_GPU_LINE_ON_L40S.md), G6's call on U2d's L40S fits: zg6 (the free GNN) against zspk on U2d's base build, re-called under U2d's zspk null on the L40S: zspk's base-build fit trained again with seeds 1 and 2, each against its seed-0 fit (N2's design, on the device the arms ran on).

# scr-gl-g6-zg6-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of zg6's p@swa minus zspk's p@swa of the same split, both U2d's base-build fits on the L40S, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of zspk's seeds 1 and 2 against its seed-0 fit on U2d's base build) (docs/SCREENS.md, section 2). D1, D2: zspk's seed-only differences on the read (U2d's zspk null). Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | +0.0102 [+0.0085, +0.0118] | -0.0003, +0.0018 | 0.0075 | GAIN | GAIN |
| L-musique | squad | in-domain | -0.0012 [-0.0033, +0.0010] | +0.0006, -0.0008 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | +0.0063 [-0.0010, +0.0131] | +0.0060, -0.0019 | 0.0089 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | +0.0030 [+0.0005, +0.0053] | -0.0030, -0.0003 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | +0.0070 [+0.0055, +0.0087] | -0.0003, +0.0012 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | -0.0795 [-0.1003, -0.0596] | -0.0071, -0.0022 | 0.0105 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | +0.0068 [+0.0054, +0.0083] | -0.0038, -0.0029 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | +0.0001 [-0.0020, +0.0023] | +0.0005, -0.0002 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | +0.0138 [+0.0070, +0.0204] | +0.0060, +0.0046 | 0.0107 | GAIN | GAIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0203 [-0.0248, -0.0161] | -0.0018, +0.0101 | 0.0145 | LOSS | LOSS |
| L-hotpotqa | 2wiki | in-domain | +0.0101 [+0.0080, +0.0123] | +0.0094, -0.0029 | 0.0139 | GAIN | **WITHIN** |
| L-hotpotqa | webqsp | zero-shot | -0.0584 [-0.0783, -0.0381] | -0.0275, +0.0079 | 0.0405 | LOSS | LOSS |

Re-call: reads with a GAIN: 2 of 12; with a LOSS: 3. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> MIXED; L-hotpotqa MIXED -> MIXED.
Changed calls: L-hotpotqa 2wiki GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
