P1c3 (docs/P1C3_KT_INPUT.md): zrck (KT input) on KIND_T's pools against zrc on today's.

# scr-p1c3-zrck-pair re-called under the seed null: **PROMISING** (filed: MIXED)

R@5 of zrck's p@swa minus zrc's p@swa of the same split (zrc's card fits on today's pools), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/P1C3_KT_INPUT.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrc R@5 (today's) | zrck R@5 (P1c3) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7782 | 0.8007 | +0.0225 [+0.0196, +0.0253] | +0.0008, -0.0023 | 0.0075 | GAIN | GAIN |
| L-musique | squad | in-domain | 0.9105 | 0.9117 | +0.0013 [-0.0008, +0.0034] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5235 | 0.5059 | -0.0176 [-0.0260, -0.0097] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9041 | 0.9063 | +0.0022 [-0.0010, +0.0057] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.8733 | 0.9365 | +0.0632 [+0.0600, +0.0666] | -0.0002, -0.0007 | 0.0075 | GAIN | GAIN |
| L-musique | webqsp | zero-shot | 0.3384 | 0.3756 | +0.0372 [+0.0220, +0.0528] | -0.0070, -0.0018 | 0.0102 | GAIN | GAIN |
| L-hotpotqa | metaqa | in-domain | 0.7793 | 0.8052 | +0.0259 [+0.0230, +0.0287] | -0.0092, -0.0051 | 0.0149 | GAIN | GAIN |
| L-hotpotqa | squad | in-domain | 0.9112 | 0.9122 | +0.0009 [-0.0013, +0.0029] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5645 | 0.5694 | +0.0050 [-0.0037, +0.0132] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8451 | 0.8564 | +0.0113 [+0.0065, +0.0160] | -0.0055, -0.0130 | 0.0200 | GAIN | **WITHIN** |
| L-hotpotqa | 2wiki | in-domain | 0.8731 | 0.9128 | +0.0397 [+0.0362, +0.0434] | -0.0020, +0.0011 | 0.0075 | GAIN | GAIN |
| L-hotpotqa | webqsp | zero-shot | 0.2865 | 0.3686 | +0.0820 [+0.0644, +0.0985] | +0.0102, +0.0129 | 0.0233 | GAIN | GAIN |

Re-call: reads with a GAIN: 6 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> PROMISING; L-hotpotqa PROMISING -> PROMISING.
Changed calls: L-musique musique LOSS -> WITHIN; L-hotpotqa hotpotqa GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
