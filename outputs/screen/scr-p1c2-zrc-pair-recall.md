P1c2 (docs/P1C2_POOLS_KIND_T.md): zrc on KIND_T's pools against zrc on today's.

# scr-p1c2-zrc-pair re-called under the seed null: **PROMISING** (filed: MIXED)

R@5 of zrc's p@swa minus zrc's p@swa of the same split (zrc's card fits on today's pools), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/P1C2_POOLS_KIND_T.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrc R@5 (today's) | zrc R@5 (P1c2) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7782 | 0.7994 | +0.0211 [+0.0183, +0.0238] | +0.0008, -0.0023 | 0.0075 | GAIN | GAIN |
| L-musique | squad | in-domain | 0.9105 | 0.9123 | +0.0019 [-0.0002, +0.0040] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5235 | 0.4789 | -0.0447 [-0.0521, -0.0372] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9041 | 0.9036 | -0.0004 [-0.0034, +0.0026] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.8733 | 0.9128 | +0.0394 [+0.0367, +0.0423] | -0.0002, -0.0007 | 0.0075 | GAIN | GAIN |
| L-musique | webqsp | zero-shot | 0.3384 | 0.3466 | +0.0082 [-0.0065, +0.0232] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.7793 | 0.8002 | +0.0209 [+0.0181, +0.0237] | -0.0092, -0.0051 | 0.0149 | GAIN | GAIN |
| L-hotpotqa | squad | in-domain | 0.9112 | 0.9134 | +0.0022 [+0.0002, +0.0043] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5645 | 0.5651 | +0.0006 [-0.0061, +0.0068] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8451 | 0.8363 | -0.0088 [-0.0132, -0.0047] | -0.0055, -0.0130 | 0.0200 | LOSS | **WITHIN** |
| L-hotpotqa | 2wiki | in-domain | 0.8731 | 0.9125 | +0.0394 [+0.0363, +0.0425] | -0.0020, +0.0011 | 0.0075 | GAIN | GAIN |
| L-hotpotqa | webqsp | zero-shot | 0.2865 | 0.3142 | +0.0276 [+0.0133, +0.0419] | +0.0102, +0.0129 | 0.0233 | GAIN | GAIN |

Re-call: reads with a GAIN: 5 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> PROMISING; L-hotpotqa MIXED -> PROMISING.
Changed calls: L-musique musique LOSS -> WITHIN; L-hotpotqa hotpotqa LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
