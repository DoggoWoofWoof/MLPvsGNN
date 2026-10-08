# scr-zkind-pair re-called under the seed null: **PROMISING** (filed: MIXED)

R@5 of zkind's p@swa minus zrm's p@swa of the same split (zrm's screen fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2 and the nineteenth round). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrm R@5 | zkind R@5 | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7789 | 0.7786 | -0.0003 [-0.0014, +0.0007] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9111 | 0.9110 | -0.0001 [-0.0019, +0.0019] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5231 | 0.5331 | +0.0100 [+0.0046, +0.0148] | +0.0503, -0.0079 | 0.0720 | GAIN | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9040 | 0.9028 | -0.0012 [-0.0034, +0.0009] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.8709 | 0.8687 | -0.0022 [-0.0040, -0.0004] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3128 | 0.3296 | +0.0168 [+0.0048, +0.0291] | -0.0070, -0.0018 | 0.0102 | GAIN | GAIN |
| L-hotpotqa | metaqa | in-domain | 0.7801 | 0.7779 | -0.0022 [-0.0033, -0.0010] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9116 | 0.9140 | +0.0024 [+0.0004, +0.0045] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5600 | 0.5677 | +0.0078 [+0.0019, +0.0139] | +0.0015, +0.0023 | 0.0075 | GAIN | GAIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8457 | 0.8344 | -0.0113 [-0.0149, -0.0078] | -0.0055, -0.0130 | 0.0200 | LOSS | **WITHIN** |
| L-hotpotqa | 2wiki | in-domain | 0.8681 | 0.8687 | +0.0007 [-0.0014, +0.0027] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2416 | 0.3007 | +0.0591 [+0.0444, +0.0741] | +0.0102, +0.0129 | 0.0233 | GAIN | GAIN |

Re-call: reads with a GAIN: 3 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique PROMISING -> PROMISING; L-hotpotqa MIXED -> PROMISING.
Changed calls: L-musique musique GAIN -> WITHIN; L-hotpotqa hotpotqa LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
