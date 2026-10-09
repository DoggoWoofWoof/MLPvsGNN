# scr-zg1-pair re-called under the seed null: **NO_GAIN** (filed: MIXED)

R@5 of zg1's p@swa minus zsp's p@swa of the same split (zsp's card fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2 and the thirty-first round). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zsp R@5 | zg1 R@5 | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7794 | 0.7783 | -0.0011 [-0.0022, +0.0000] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9125 | 0.9127 | +0.0003 [-0.0017, +0.0023] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5111 | 0.5188 | +0.0077 [+0.0017, +0.0136] | +0.0503, -0.0079 | 0.0720 | GAIN | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9065 | 0.9071 | +0.0005 [-0.0016, +0.0028] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.8724 | 0.8787 | +0.0062 [+0.0043, +0.0082] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3245 | 0.3089 | -0.0156 [-0.0250, -0.0068] | -0.0070, -0.0018 | 0.0102 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | 0.7807 | 0.7783 | -0.0024 [-0.0036, -0.0012] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9109 | 0.9119 | +0.0010 [-0.0011, +0.0032] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5647 | 0.5677 | +0.0030 [-0.0027, +0.0090] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8435 | 0.8485 | +0.0050 [+0.0012, +0.0086] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.8744 | 0.8816 | +0.0072 [+0.0048, +0.0094] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2262 | 0.2285 | +0.0023 [-0.0075, +0.0119] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
