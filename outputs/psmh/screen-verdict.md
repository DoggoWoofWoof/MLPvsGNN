# PSMH, the hard gap's screen on Kaggle's T4: **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when, under WQK's floors on zrckp, no read of the twelve is LOSS and at least one is GAIN; NOT_ADOPTED otherwise; INCOMPLETE when a comparison or the floors are missing, not WQK's or PSMH's, or not twelve reads (docs/PSMH_HARD_GAP_ON_T4.md, section 4).

zrckph (zrckp with each fixed column's hard gap below the pool's best) against WQK's seed-0 zrckp fit of each split on the same T4, every read called under WQK's floors on zrckp: 0 GAIN (none), 2 LOSS (WQ-musique webqsp, WQ-hotpotqa webqsp).

Next: zrckpg1: PSM's soft gap divided by 10, in [-1, 0], the scale of the z-scores beside it.

FC@5 and hit@1 are beside, deciding nothing.

| split | dataset | read | base R@5 | new R@5 | delta R@5 [95% CI] | filed | floor | call | delta FC@5 | delta hit@1 |
|---|---|---|---:|---:|---|---|---:|---|---:|---:|
| WQ-musique | metaqa | zero-shot | 0.4393 | 0.4013 | -0.0380 [-0.0452, -0.0309] | LOSS | 0.0621 | WITHIN | -0.0363 | -0.0655 |
| WQ-musique | squad | in-domain | 0.9106 | 0.9094 | -0.0012 [-0.0032, +0.0007] | WITHIN | 0.0075 | WITHIN | -0.0012 | -0.0012 |
| WQ-musique | musique | zero-shot | 0.4972 | 0.5049 | +0.0077 [+0.0012, +0.0139] | GAIN | 0.0566 | WITHIN | +0.0074 | +0.0790 |
| WQ-musique | hotpotqa | in-domain | 0.9051 | 0.8989 | -0.0063 [-0.0088, -0.0038] | WITHIN | 0.0087 | WITHIN | -0.0124 | -0.0150 |
| WQ-musique | 2wiki | in-domain | 0.9315 | 0.9244 | -0.0072 [-0.0089, -0.0054] | WITHIN | 0.0093 | WITHIN | -0.0250 | +0.0038 |
| WQ-musique | webqsp | in-domain | 0.6433 | 0.6251 | -0.0182 [-0.0279, -0.0090] | LOSS | 0.0075 | **LOSS** | -0.0240 | -0.0266 |
| WQ-hotpotqa | metaqa | zero-shot | 0.3927 | 0.4282 | +0.0355 [+0.0296, +0.0411] | GAIN | 0.0489 | WITHIN | +0.0289 | +0.0017 |
| WQ-hotpotqa | squad | in-domain | 0.9117 | 0.9128 | +0.0011 [-0.0010, +0.0033] | WITHIN | 0.0075 | WITHIN | +0.0011 | -0.0001 |
| WQ-hotpotqa | musique | in-domain | 0.5743 | 0.5760 | +0.0017 [-0.0030, +0.0065] | WITHIN | 0.0075 | WITHIN | +0.0033 | +0.0012 |
| WQ-hotpotqa | hotpotqa | zero-shot | 0.8525 | 0.8469 | -0.0056 [-0.0090, -0.0022] | WITHIN | 0.0089 | WITHIN | -0.0070 | -0.0192 |
| WQ-hotpotqa | 2wiki | in-domain | 0.9299 | 0.9264 | -0.0036 [-0.0052, -0.0018] | WITHIN | 0.0075 | WITHIN | -0.0088 | -0.0017 |
| WQ-hotpotqa | webqsp | in-domain | 0.6382 | 0.6243 | -0.0139 [-0.0225, -0.0054] | LOSS | 0.0075 | **LOSS** | -0.0193 | -0.0126 |
