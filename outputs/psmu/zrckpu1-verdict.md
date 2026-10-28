# PSMU, the lift divided by 10 (zrckpu1), on Kaggle's T4: **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when, under WQK's floors on zrckp, no read of the twelve is LOSS and at least one is GAIN; NOT_ADOPTED otherwise; INCOMPLETE when a comparison or the floors are missing, not WQK's or PSMU's, or not twelve reads (docs/PSMU_UNIFORM_LIFT_ON_T4.md, section 4; each arm its own verdict).

zrckpu1 (zrckp with each fixed column's lift above a uniform pool, divided by 10) against WQK's seed-0 zrckp fit of each split on the same T4, every read called under WQK's floors on zrckp: 0 GAIN (none), 0 LOSS (none).

Next: with zrckpu's verdict (PSMU's section 5): zrckpu's next if it is adopted; if neither arm is adopted, the pool-softmax family closes for the MLP's input: the soft gap, the hard gap and the lift, each undivided and divided by 10, all NOT_ADOPTED under WQK's floors, so zrckp's input stays; the next forms under WQ go on, each in its own declared file: G6r under WQ on the T4 (WQK's section 9, item 1) and U2n's zrckpt refit on U2s's pick under WQ (WQP's section 7).

FC@5 and hit@1 are beside, deciding nothing.

| split | dataset | read | base R@5 | new R@5 | delta R@5 [95% CI] | filed | floor | call | delta FC@5 | delta hit@1 |
|---|---|---|---:|---:|---|---|---:|---|---:|---:|
| WQ-musique | metaqa | zero-shot | 0.4393 | 0.4361 | -0.0032 [-0.0082, +0.0018] | WITHIN | 0.0621 | WITHIN | +0.0007 | -0.0226 |
| WQ-musique | squad | in-domain | 0.9106 | 0.9097 | -0.0008 [-0.0027, +0.0009] | WITHIN | 0.0075 | WITHIN | -0.0008 | -0.0015 |
| WQ-musique | musique | zero-shot | 0.4972 | 0.4936 | -0.0036 [-0.0090, +0.0018] | WITHIN | 0.0566 | WITHIN | -0.0050 | +0.0157 |
| WQ-musique | hotpotqa | in-domain | 0.9051 | 0.9032 | -0.0019 [-0.0040, +0.0002] | WITHIN | 0.0087 | WITHIN | -0.0035 | -0.0107 |
| WQ-musique | 2wiki | in-domain | 0.9315 | 0.9314 | -0.0001 [-0.0015, +0.0013] | WITHIN | 0.0093 | WITHIN | -0.0010 | -0.0019 |
| WQ-musique | webqsp | in-domain | 0.6433 | 0.6388 | -0.0046 [-0.0133, +0.0040] | WITHIN | 0.0075 | WITHIN | -0.0040 | -0.0140 |
| WQ-hotpotqa | metaqa | zero-shot | 0.3927 | 0.4342 | +0.0415 [+0.0363, +0.0467] | GAIN | 0.0489 | WITHIN | +0.0358 | +0.0487 |
| WQ-hotpotqa | squad | in-domain | 0.9117 | 0.9127 | +0.0010 [-0.0009, +0.0029] | WITHIN | 0.0075 | WITHIN | +0.0010 | -0.0002 |
| WQ-hotpotqa | musique | in-domain | 0.5743 | 0.5731 | -0.0012 [-0.0062, +0.0037] | WITHIN | 0.0075 | WITHIN | -0.0041 | -0.0050 |
| WQ-hotpotqa | hotpotqa | zero-shot | 0.8525 | 0.8568 | +0.0043 [+0.0014, +0.0074] | WITHIN | 0.0089 | WITHIN | +0.0069 | -0.0023 |
| WQ-hotpotqa | 2wiki | in-domain | 0.9299 | 0.9297 | -0.0003 [-0.0018, +0.0013] | WITHIN | 0.0075 | WITHIN | -0.0006 | +0.0009 |
| WQ-hotpotqa | webqsp | in-domain | 0.6382 | 0.6315 | -0.0067 [-0.0141, +0.0004] | WITHIN | 0.0075 | WITHIN | -0.0093 | +0.0007 |
