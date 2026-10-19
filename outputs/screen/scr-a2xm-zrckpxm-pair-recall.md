A2xm (docs/A2XM_ZTOP50_REPLACE.md): zrckpxm (zrckp with zret's reference cut to the pool's top 50 retrieved rows by rrf (ztop50's form)) against zrckp.

# scr-a2xm-zrckpxm-pair re-called under the seed null: **NO_GAIN** (filed: MIXED)

R@5 of zrckpxm's p@swa minus zrckp's p@swa of the same split (zrckp's card fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/A2XM_ZTOP50_REPLACE.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrckp R@5 | zrckpxm R@5 (A2xm's top-50 reference) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8019 | 0.8002 | -0.0017 [-0.0027, -0.0007] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9116 | 0.9122 | +0.0006 [-0.0013, +0.0025] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5121 | 0.5235 | +0.0113 [+0.0053, +0.0173] | +0.0503, -0.0079 | 0.0720 | GAIN | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9065 | 0.9040 | -0.0026 [-0.0051, -0.0003] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 | 0.9367 | +0.0020 [+0.0005, +0.0036] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 | 0.3661 | +0.0034 [-0.0082, +0.0151] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8044 | 0.8032 | -0.0012 [-0.0021, -0.0002] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 | 0.9118 | -0.0009 [-0.0029, +0.0011] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 | 0.5760 | +0.0016 [-0.0042, +0.0072] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 | 0.8520 | +0.0007 [-0.0026, +0.0039] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 | 0.9361 | +0.0011 [-0.0005, +0.0027] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 | 0.3124 | -0.0351 [-0.0494, -0.0208] | +0.0102, +0.0129 | 0.0233 | LOSS | LOSS |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique PROMISING -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
