# scr-zgr-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of zgr's p@swa minus zrm's p@swa of the same split (zrm's screen fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2 and the twenty-seventh round). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrm R@5 | zgr R@5 | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7789 | 0.7823 | +0.0034 [+0.0020, +0.0047] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9111 | 0.8993 | -0.0118 [-0.0149, -0.0086] | +0.0001, +0.0013 | 0.0075 | LOSS | LOSS |
| L-musique | musique | zero-shot | 0.5231 | 0.5174 | -0.0057 [-0.0122, +0.0007] | +0.0503, -0.0079 | 0.0720 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | 0.9040 | 0.8977 | -0.0063 [-0.0091, -0.0033] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.8709 | 0.8706 | -0.0003 [-0.0024, +0.0018] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3128 | 0.3124 | -0.0004 [-0.0133, +0.0116] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.7801 | 0.7826 | +0.0026 [+0.0013, +0.0039] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9116 | 0.8975 | -0.0141 [-0.0171, -0.0109] | +0.0002, +0.0007 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | musique | in-domain | 0.5600 | 0.5125 | -0.0474 [-0.0556, -0.0393] | +0.0015, +0.0023 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | hotpotqa | zero-shot | 0.8457 | 0.8194 | -0.0263 [-0.0308, -0.0221] | -0.0055, -0.0130 | 0.0200 | LOSS | LOSS |
| L-hotpotqa | 2wiki | in-domain | 0.8681 | 0.8656 | -0.0025 [-0.0049, +0.0001] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2416 | 0.2350 | -0.0065 [-0.0199, +0.0068] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 4. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: none.
Within 0.00015 of the floor (the call stands): none.
