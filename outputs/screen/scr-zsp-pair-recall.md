# scr-zsp-pair re-called under the seed null: **PROMISING** (filed: PROMISING)

R@5 of zsp's p@swa minus zrm's p@swa of the same split (zrm's CPU fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2 and the twenty-third round). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrm (CPU) R@5 | zsp R@5 | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7780 | 0.7786 | +0.0006 [-0.0005, +0.0017] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9118 | 0.9100 | -0.0018 [-0.0039, +0.0003] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5126 | 0.5131 | +0.0004 [-0.0051, +0.0057] | +0.0503, -0.0079 | 0.0720 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | 0.9038 | 0.9051 | +0.0013 [-0.0011, +0.0038] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.8704 | 0.8756 | +0.0052 [+0.0033, +0.0073] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3212 | 0.3247 | +0.0035 [-0.0064, +0.0131] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.7787 | 0.7800 | +0.0013 [+0.0002, +0.0024] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9114 | 0.9148 | +0.0035 [+0.0014, +0.0056] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5714 | 0.5748 | +0.0034 [-0.0024, +0.0091] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8481 | 0.8406 | -0.0074 [-0.0110, -0.0040] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.8663 | 0.8763 | +0.0100 [+0.0081, +0.0123] | -0.0020, +0.0011 | 0.0075 | GAIN | GAIN |
| L-hotpotqa | webqsp | zero-shot | 0.2364 | 0.2333 | -0.0030 [-0.0122, +0.0064] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 1 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa PROMISING -> PROMISING.
Changed calls: none.
Within 0.00015 of the floor (the call stands): none.
