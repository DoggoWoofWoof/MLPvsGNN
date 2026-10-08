# scr-zrct-pair re-called under the seed null: **PROMISING** (filed: PROMISING)

R@5 of zrc's p@swa minus zrm's p@swa of the same split (zrm's screen fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2 and the eighteenth round). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrm R@5 | zrc R@5 | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7789 | 0.7782 | -0.0007 [-0.0017, +0.0003] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9111 | 0.9105 | -0.0006 [-0.0024, +0.0013] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5231 | 0.5235 | +0.0004 [-0.0042, +0.0050] | +0.0503, -0.0079 | 0.0720 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | 0.9040 | 0.9041 | +0.0001 [-0.0020, +0.0020] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.8709 | 0.8733 | +0.0024 [+0.0007, +0.0041] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3128 | 0.3384 | +0.0257 [+0.0108, +0.0402] | -0.0070, -0.0018 | 0.0102 | GAIN | GAIN |
| L-hotpotqa | metaqa | in-domain | 0.7801 | 0.7793 | -0.0008 [-0.0017, +0.0001] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9116 | 0.9112 | -0.0003 [-0.0023, +0.0014] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5600 | 0.5645 | +0.0045 [-0.0006, +0.0098] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8457 | 0.8451 | -0.0006 [-0.0035, +0.0023] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.8681 | 0.8731 | +0.0051 [+0.0033, +0.0069] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2416 | 0.2865 | +0.0450 [+0.0310, +0.0586] | +0.0102, +0.0129 | 0.0233 | GAIN | GAIN |

Re-call: reads with a GAIN: 2 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique PROMISING -> PROMISING; L-hotpotqa PROMISING -> PROMISING.
Changed calls: none.
Within 0.00015 of the floor (the call stands): none.
