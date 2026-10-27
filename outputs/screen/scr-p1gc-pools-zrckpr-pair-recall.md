P1g-c (docs/P1GC_ADOPTED_ON_P1GB.md), the pools call: zrckpr (R1's MLP as filed) on P1g-b's pools against zrckpr on P1c2's pools.

# scr-p1gc-pools-zrckpr-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of zrckpr's p@swa minus zrckpr's p@swa of the same split (zrckpr's card fits on P1c2's pools), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/P1GC_ADOPTED_ON_P1GB.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrckpr R@5 (P1c2's) | zrckpr R@5 (P1g-b's) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8012 | 0.8004 | -0.0008 [-0.0024, +0.0008] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9127 | 0.9135 | +0.0008 [-0.0010, +0.0026] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5197 | 0.5053 | -0.0144 [-0.0199, -0.0085] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9083 | 0.9059 | -0.0024 [-0.0049, +0.0001] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9361 | 0.9365 | +0.0004 [-0.0012, +0.0019] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3762 | 0.3873 | +0.0111 [-0.0002, +0.0218] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8040 | 0.8012 | -0.0029 [-0.0045, -0.0013] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9114 | 0.9112 | -0.0002 [-0.0021, +0.0019] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5779 | 0.5689 | -0.0090 [-0.0148, -0.0036] | +0.0015, +0.0023 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | hotpotqa | zero-shot | 0.8558 | 0.8519 | -0.0039 [-0.0072, -0.0007] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9331 | 0.9303 | -0.0028 [-0.0046, -0.0011] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3411 | 0.3370 | -0.0041 [-0.0172, +0.0093] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
