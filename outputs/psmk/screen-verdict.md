# PSMK, PSM's screen on Kaggle's T4: **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when, under WQK's floors on zrckp, no read of the twelve is LOSS and at least one is GAIN; NOT_ADOPTED otherwise; INCOMPLETE when a comparison or the floors are missing, not WQK's or PSMK's, or not twelve reads (docs/PSMK_PSM_ON_T4.md, section 3).

zrckpg (zrckp with each fixed column's pool gap) against WQK's seed-0 zrckp fit of each split on the same T4, every read called under WQK's floors on zrckp: 0 GAIN (none), 7 LOSS (WQ-musique hotpotqa, WQ-musique 2wiki, WQ-musique webqsp, WQ-hotpotqa musique, WQ-hotpotqa hotpotqa, WQ-hotpotqa 2wiki, WQ-hotpotqa webqsp).

Next: zrckph: the hard gap, z_i minus the pool's largest z, floored at -10.

FC@5 and hit@1 are beside, deciding nothing.

| split | dataset | read | base R@5 | new R@5 | delta R@5 [95% CI] | filed | floor | call | delta FC@5 | delta hit@1 |
|---|---|---|---:|---:|---|---|---:|---|---:|---:|
| WQ-musique | metaqa | zero-shot | 0.4393 | 0.4686 | +0.0293 [+0.0236, +0.0350] | GAIN | 0.0621 | WITHIN | +0.0199 | -0.0144 |
| WQ-musique | squad | in-domain | 0.9106 | 0.9079 | -0.0027 [-0.0049, -0.0007] | WITHIN | 0.0075 | WITHIN | -0.0027 | -0.0041 |
| WQ-musique | musique | zero-shot | 0.4972 | 0.5048 | +0.0076 [+0.0006, +0.0142] | GAIN | 0.0566 | WITHIN | +0.0000 | +0.0856 |
| WQ-musique | hotpotqa | in-domain | 0.9051 | 0.8797 | -0.0254 [-0.0286, -0.0221] | LOSS | 0.0087 | **LOSS** | -0.0459 | -0.0404 |
| WQ-musique | 2wiki | in-domain | 0.9315 | 0.9200 | -0.0115 [-0.0133, -0.0097] | LOSS | 0.0093 | **LOSS** | -0.0328 | +0.0150 |
| WQ-musique | webqsp | in-domain | 0.6433 | 0.6300 | -0.0133 [-0.0225, -0.0036] | LOSS | 0.0075 | **LOSS** | -0.0153 | -0.0439 |
| WQ-hotpotqa | metaqa | zero-shot | 0.3927 | 0.3953 | +0.0026 [-0.0038, +0.0091] | WITHIN | 0.0489 | WITHIN | +0.0032 | -0.0217 |
| WQ-hotpotqa | squad | in-domain | 0.9117 | 0.9068 | -0.0049 [-0.0072, -0.0026] | WITHIN | 0.0075 | WITHIN | -0.0049 | -0.0024 |
| WQ-hotpotqa | musique | in-domain | 0.5743 | 0.5568 | -0.0175 [-0.0238, -0.0112] | LOSS | 0.0075 | **LOSS** | -0.0248 | -0.0128 |
| WQ-hotpotqa | hotpotqa | zero-shot | 0.8525 | 0.8379 | -0.0145 [-0.0184, -0.0106] | LOSS | 0.0089 | **LOSS** | -0.0236 | -0.0224 |
| WQ-hotpotqa | 2wiki | in-domain | 0.9299 | 0.9098 | -0.0201 [-0.0223, -0.0179] | LOSS | 0.0075 | **LOSS** | -0.0431 | +0.0082 |
| WQ-hotpotqa | webqsp | in-domain | 0.6382 | 0.6240 | -0.0142 [-0.0240, -0.0042] | LOSS | 0.0075 | **LOSS** | -0.0220 | -0.0193 |
