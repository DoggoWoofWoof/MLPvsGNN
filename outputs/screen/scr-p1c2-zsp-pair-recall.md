P1c2 (docs/P1C2_POOLS_KIND_T.md): zsp on KIND_T's pools against zsp on today's.

# scr-p1c2-zsp-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of zsp's p@swa minus zsp's p@swa of the same split (zsp's card fits on today's pools), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/P1C2_POOLS_KIND_T.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zsp R@5 (today's) | zsp R@5 (P1c2) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7794 | 0.8008 | +0.0214 [+0.0185, +0.0243] | +0.0008, -0.0023 | 0.0075 | GAIN | GAIN |
| L-musique | squad | in-domain | 0.9125 | 0.9119 | -0.0006 [-0.0027, +0.0015] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5111 | 0.4792 | -0.0320 [-0.0392, -0.0242] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9065 | 0.9062 | -0.0003 [-0.0032, +0.0026] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.8724 | 0.9174 | +0.0450 [+0.0422, +0.0478] | -0.0002, -0.0007 | 0.0075 | GAIN | GAIN |
| L-musique | webqsp | zero-shot | 0.3245 | 0.3078 | -0.0167 [-0.0333, -0.0003] | -0.0070, -0.0018 | 0.0102 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | 0.7807 | 0.8008 | +0.0201 [+0.0172, +0.0228] | -0.0092, -0.0051 | 0.0149 | GAIN | GAIN |
| L-hotpotqa | squad | in-domain | 0.9109 | 0.9111 | +0.0002 [-0.0020, +0.0025] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5647 | 0.5707 | +0.0060 [-0.0007, +0.0126] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8435 | 0.8475 | +0.0040 [-0.0001, +0.0080] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.8744 | 0.9171 | +0.0426 [+0.0395, +0.0458] | -0.0020, +0.0011 | 0.0075 | GAIN | GAIN |
| L-hotpotqa | webqsp | zero-shot | 0.2262 | 0.2603 | +0.0341 [+0.0181, +0.0493] | +0.0102, +0.0129 | 0.0233 | GAIN | GAIN |

Re-call: reads with a GAIN: 5 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> MIXED; L-hotpotqa PROMISING -> PROMISING.
Changed calls: L-musique musique LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
