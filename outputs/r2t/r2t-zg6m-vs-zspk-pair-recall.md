R2t (docs/R2T_BRIDGE_EM_ON_T4.md): zg6m (zg6 + R2's bridge head on its final score) against KG's zspk on the T4, re-called under KG's zspk null on the T4: zspk's base-build fit trained again with seeds 1 and 2, each against its seed-0 fit, beside, deciding nothing.

# r2t-zg6m-vs-zspk-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of zg6m's p@swa minus KG's seed-0 zspk's p@swa of the same split, both on U2d's base build on the T4, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of zspk's seeds 1 and 2 against its seed-0 fit on the T4) (docs/SCREENS.md, section 2). D1, D2: zspk's seed-only differences on the read (KG's zspk null). Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | +0.0110 [+0.0092, +0.0126] | +0.0025, +0.0031 | 0.0075 | GAIN | GAIN |
| L-musique | squad | in-domain | +0.0003 [-0.0021, +0.0028] | +0.0007, +0.0012 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | +0.0098 [+0.0028, +0.0168] | -0.0066, -0.0072 | 0.0138 | GAIN | **WITHIN** |
| L-musique | hotpotqa | in-domain | +0.0057 [+0.0035, +0.0081] | -0.0011, -0.0018 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | +0.0053 [+0.0037, +0.0069] | -0.0030, -0.0011 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | -0.0969 [-0.1184, -0.0747] | +0.0009, +0.0007 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | +0.0076 [+0.0059, +0.0093] | -0.0027, -0.0011 | 0.0075 | GAIN | GAIN |
| L-hotpotqa | squad | in-domain | -0.0060 [-0.0084, -0.0036] | +0.0001, -0.0001 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | +0.0158 [+0.0089, +0.0224] | +0.0008, +0.0074 | 0.0105 | GAIN | GAIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0169 [-0.0209, -0.0128] | -0.0047, +0.0001 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | 2wiki | in-domain | +0.0168 [+0.0147, +0.0189] | +0.0114, +0.0105 | 0.0219 | GAIN | **WITHIN** |
| L-hotpotqa | webqsp | zero-shot | -0.0815 [-0.1026, -0.0611] | -0.0166, +0.0041 | 0.0242 | LOSS | LOSS |

Re-call: reads with a GAIN: 3 of 12; with a LOSS: 3. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> MIXED; L-hotpotqa MIXED -> MIXED.
Changed calls: L-musique musique GAIN -> WITHIN; L-hotpotqa 2wiki GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
