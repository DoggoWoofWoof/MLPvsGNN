# PSMS, the hard gap divided by 10 (zrckph1), on Kaggle's T4: **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when, under WQK's floors on zrckp, no read of the twelve is LOSS and at least one is GAIN; NOT_ADOPTED otherwise; INCOMPLETE when a comparison or the floors are missing, not WQK's or PSMS's, or not twelve reads (docs/PSMS_GAPS_AT_Z_SCALE_ON_T4.md, section 4; each arm its own verdict).

zrckph1 (zrckp with each fixed column's hard gap below the pool's best, divided by 10) against WQK's seed-0 zrckp fit of each split on the same T4, every read called under WQK's floors on zrckp: 0 GAIN (none), 0 LOSS (none).

Next: with zrckpg1's verdict (PSMS's section 5): zrckpg1's next if it is adopted; if neither arm is adopted, zrckpu1, PSM's section 6 form for a GAIN and a LOSS (the row's lift above a uniform pool, log(N_q p_i) floored at 0) divided by 10.

FC@5 and hit@1 are beside, deciding nothing.

| split | dataset | read | base R@5 | new R@5 | delta R@5 [95% CI] | filed | floor | call | delta FC@5 | delta hit@1 |
|---|---|---|---:|---:|---|---|---:|---|---:|---:|
| WQ-musique | metaqa | zero-shot | 0.4393 | 0.4250 | -0.0144 [-0.0195, -0.0089] | LOSS | 0.0621 | WITHIN | -0.0174 | -0.0264 |
| WQ-musique | squad | in-domain | 0.9106 | 0.9114 | +0.0008 [-0.0010, +0.0026] | WITHIN | 0.0075 | WITHIN | +0.0008 | +0.0003 |
| WQ-musique | musique | zero-shot | 0.4972 | 0.4997 | +0.0025 [-0.0032, +0.0082] | WITHIN | 0.0566 | WITHIN | -0.0029 | +0.0381 |
| WQ-musique | hotpotqa | in-domain | 0.9051 | 0.9022 | -0.0030 [-0.0053, -0.0007] | WITHIN | 0.0087 | WITHIN | -0.0069 | -0.0018 |
| WQ-musique | 2wiki | in-domain | 0.9315 | 0.9306 | -0.0009 [-0.0024, +0.0007] | WITHIN | 0.0093 | WITHIN | -0.0056 | +0.0056 |
| WQ-musique | webqsp | in-domain | 0.6433 | 0.6410 | -0.0023 [-0.0109, +0.0060] | WITHIN | 0.0075 | WITHIN | -0.0033 | -0.0120 |
| WQ-hotpotqa | metaqa | zero-shot | 0.3927 | 0.4104 | +0.0177 [+0.0128, +0.0230] | GAIN | 0.0489 | WITHIN | +0.0145 | +0.0069 |
| WQ-hotpotqa | squad | in-domain | 0.9117 | 0.9111 | -0.0006 [-0.0024, +0.0013] | WITHIN | 0.0075 | WITHIN | -0.0006 | -0.0012 |
| WQ-hotpotqa | musique | in-domain | 0.5743 | 0.5733 | -0.0010 [-0.0059, +0.0038] | WITHIN | 0.0075 | WITHIN | -0.0021 | +0.0025 |
| WQ-hotpotqa | hotpotqa | zero-shot | 0.8525 | 0.8519 | -0.0005 [-0.0036, +0.0025] | WITHIN | 0.0089 | WITHIN | -0.0005 | -0.0178 |
| WQ-hotpotqa | 2wiki | in-domain | 0.9299 | 0.9303 | +0.0004 [-0.0012, +0.0019] | WITHIN | 0.0075 | WITHIN | +0.0008 | -0.0078 |
| WQ-hotpotqa | webqsp | in-domain | 0.6382 | 0.6330 | -0.0053 [-0.0127, +0.0017] | WITHIN | 0.0075 | WITHIN | -0.0106 | -0.0153 |
