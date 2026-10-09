# scr-zdp-pair re-called under the seed null: **NO_GAIN** (filed: MIXED)

R@5 of zdp's p@swa minus zsp's p@swa of the same split (zsp's CPU screen fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2 and the thirtieth round). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zsp R@5 | zdp R@5 | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7786 | 0.7796 | +0.0009 [+0.0001, +0.0019] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9100 | 0.9136 | +0.0035 [+0.0015, +0.0056] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5131 | 0.5102 | -0.0029 [-0.0076, +0.0021] | +0.0503, -0.0079 | 0.0720 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | 0.9051 | 0.9041 | -0.0010 [-0.0033, +0.0014] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.8756 | 0.8755 | -0.0001 [-0.0019, +0.0017] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3247 | 0.3114 | -0.0133 [-0.0222, -0.0045] | -0.0070, -0.0018 | 0.0102 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | 0.7800 | 0.7801 | +0.0001 [-0.0010, +0.0012] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9148 | 0.9135 | -0.0013 [-0.0034, +0.0007] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5748 | 0.5677 | -0.0071 [-0.0130, -0.0013] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8406 | 0.8391 | -0.0016 [-0.0049, +0.0019] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.8763 | 0.8738 | -0.0025 [-0.0045, -0.0005] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2333 | 0.2507 | +0.0173 [+0.0078, +0.0274] | +0.0102, +0.0129 | 0.0233 | GAIN | **WITHIN** |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa PROMISING -> NO_GAIN.
Changed calls: L-hotpotqa webqsp GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
