R2t (docs/R2T_BRIDGE_EM_ON_T4.md): zg6m (zg6 + R2's bridge head on its final score) against KG's zg6 on the T4, re-called under KG's zspk null on the T4, each read's seed-0 R@5 moved to zg6's through KG's compares of zg6 with zspk (D1, D2 and the floors stay zspk's) (re-based through scr-kg-zg6-vs-zspk, scr-kg-zg6-vs-zspk-hp), deciding.

# r2t-zg6m-pair re-called under the seed null: **NO_GAIN** (filed: MIXED)

R@5 of zg6m's p@swa minus KG's seed-0 zg6's p@swa of the same split, both on U2d's base build on the T4, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of zspk's seeds 1 and 2 against its seed-0 fit on the T4) (docs/SCREENS.md, section 2). D1, D2: zspk's seed-only differences on the read (KG's zspk null). Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | -0.0004 [-0.0015, +0.0007] | +0.0025, +0.0031 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | -0.0014 [-0.0035, +0.0006] | +0.0007, +0.0012 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | +0.0107 [+0.0053, +0.0163] | -0.0066, -0.0072 | 0.0138 | GAIN | **WITHIN** |
| L-musique | hotpotqa | in-domain | +0.0016 [-0.0003, +0.0037] | -0.0011, -0.0018 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | -0.0031 [-0.0045, -0.0017] | -0.0030, -0.0011 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | -0.0216 [-0.0393, -0.0047] | +0.0009, +0.0007 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | -0.0002 [-0.0013, +0.0009] | -0.0027, -0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | -0.0043 [-0.0066, -0.0020] | +0.0001, -0.0001 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | -0.0041 [-0.0100, +0.0024] | +0.0008, +0.0074 | 0.0105 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0011 [-0.0043, +0.0022] | -0.0047, +0.0001 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | -0.0030 [-0.0045, -0.0014] | +0.0114, +0.0105 | 0.0219 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | -0.0605 [-0.0778, -0.0424] | -0.0166, +0.0041 | 0.0242 | LOSS | LOSS |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
