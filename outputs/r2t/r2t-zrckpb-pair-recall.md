R2t (docs/R2T_BRIDGE_EM_ON_T4.md): zrckpb (zrckp + R2's bridge head) against KG's zrckp on the T4, re-called under KG's zrckp null on the T4: zrckp's base-build fit trained again with seeds 1 and 2, each against its seed-0 fit, deciding.

# r2t-zrckpb-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of zrckpb's p@swa minus KG's seed-0 zrckp's p@swa of the same split, both on U2d's base build on the T4, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of zrckp's seeds 1 and 2 against its seed-0 fit on the T4) (docs/SCREENS.md, section 2). D1, D2: zrckp's seed-only differences on the read (KG's zrckp null). Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | +0.0054 [+0.0042, +0.0067] | +0.0041, +0.0043 | 0.0084 | WITHIN | WITHIN |
| L-musique | squad | in-domain | -0.0010 [-0.0031, +0.0010] | -0.0005, -0.0009 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | +0.0091 [+0.0035, +0.0149] | -0.0015, -0.0012 | 0.0075 | GAIN | GAIN |
| L-musique | hotpotqa | in-domain | +0.0022 [-0.0003, +0.0048] | +0.0031, +0.0014 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | +0.0026 [+0.0011, +0.0041] | +0.0055, +0.0062 | 0.0117 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | +0.0120 [+0.0014, +0.0227] | -0.0047, -0.0121 | 0.0184 | GAIN | **WITHIN** |
| L-hotpotqa | metaqa | in-domain | +0.0018 [+0.0006, +0.0031] | -0.0024, -0.0014 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | -0.0033 [-0.0056, -0.0011] | -0.0009, +0.0010 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | -0.0133 [-0.0199, -0.0068] | +0.0085, -0.0030 | 0.0127 | LOSS | LOSS |
| L-hotpotqa | hotpotqa | zero-shot | -0.0050 [-0.0083, -0.0016] | +0.0058, +0.0012 | 0.0084 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | -0.0014 [-0.0030, +0.0002] | -0.0002, -0.0000 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | +0.0098 [-0.0020, +0.0212] | +0.0126, +0.0110 | 0.0237 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 1 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique PROMISING -> PROMISING; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique webqsp GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
