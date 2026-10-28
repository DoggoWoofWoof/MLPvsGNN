# PSMS, the soft gap divided by 10 (zrckpg1), on Kaggle's T4: **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when, under WQK's floors on zrckp, no read of the twelve is LOSS and at least one is GAIN; NOT_ADOPTED otherwise; INCOMPLETE when a comparison or the floors are missing, not WQK's or PSMS's, or not twelve reads (docs/PSMS_GAPS_AT_Z_SCALE_ON_T4.md, section 4; each arm its own verdict).

zrckpg1 (zrckp with each fixed column's pool gap divided by 10) against WQK's seed-0 zrckp fit of each split on the same T4, every read called under WQK's floors on zrckp: 0 GAIN (none), 2 LOSS (WQ-hotpotqa hotpotqa, WQ-hotpotqa webqsp).

Next: with zrckph1's verdict (PSMS's section 5): zrckph1's next if it is adopted; if neither arm is adopted, zrckpu1, PSM's section 6 form for a GAIN and a LOSS (the row's lift above a uniform pool, log(N_q p_i) floored at 0) divided by 10.

FC@5 and hit@1 are beside, deciding nothing.

| split | dataset | read | base R@5 | new R@5 | delta R@5 [95% CI] | filed | floor | call | delta FC@5 | delta hit@1 |
|---|---|---|---:|---:|---|---|---:|---|---:|---:|
| WQ-musique | metaqa | zero-shot | 0.4393 | 0.4617 | +0.0224 [+0.0172, +0.0273] | GAIN | 0.0621 | WITHIN | +0.0107 | +0.0321 |
| WQ-musique | squad | in-domain | 0.9106 | 0.9090 | -0.0015 [-0.0034, +0.0003] | WITHIN | 0.0075 | WITHIN | -0.0015 | +0.0007 |
| WQ-musique | musique | zero-shot | 0.4972 | 0.4613 | -0.0359 [-0.0427, -0.0294] | LOSS | 0.0566 | WITHIN | -0.0294 | -0.0319 |
| WQ-musique | hotpotqa | in-domain | 0.9051 | 0.9007 | -0.0044 [-0.0067, -0.0022] | WITHIN | 0.0087 | WITHIN | -0.0082 | -0.0182 |
| WQ-musique | 2wiki | in-domain | 0.9315 | 0.9287 | -0.0028 [-0.0043, -0.0013] | WITHIN | 0.0093 | WITHIN | -0.0096 | +0.0031 |
| WQ-musique | webqsp | in-domain | 0.6433 | 0.6353 | -0.0080 [-0.0164, +0.0002] | WITHIN | 0.0075 | WITHIN | -0.0093 | -0.0047 |
| WQ-hotpotqa | metaqa | zero-shot | 0.3927 | 0.4110 | +0.0183 [+0.0132, +0.0237] | GAIN | 0.0489 | WITHIN | +0.0137 | +0.0033 |
| WQ-hotpotqa | squad | in-domain | 0.9117 | 0.9115 | -0.0003 [-0.0023, +0.0018] | WITHIN | 0.0075 | WITHIN | -0.0003 | +0.0013 |
| WQ-hotpotqa | musique | in-domain | 0.5743 | 0.5735 | -0.0008 [-0.0057, +0.0042] | WITHIN | 0.0075 | WITHIN | -0.0021 | +0.0041 |
| WQ-hotpotqa | hotpotqa | zero-shot | 0.8525 | 0.8409 | -0.0116 [-0.0148, -0.0085] | LOSS | 0.0089 | **LOSS** | -0.0228 | -0.0169 |
| WQ-hotpotqa | 2wiki | in-domain | 0.9299 | 0.9281 | -0.0018 [-0.0035, -0.0001] | WITHIN | 0.0075 | WITHIN | -0.0025 | -0.0038 |
| WQ-hotpotqa | webqsp | in-domain | 0.6382 | 0.6277 | -0.0105 [-0.0187, -0.0022] | LOSS | 0.0075 | **LOSS** | -0.0153 | -0.0220 |
