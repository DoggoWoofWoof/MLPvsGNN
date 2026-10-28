# PSMU, the lift above a uniform pool (zrckpu), on Kaggle's T4: **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when, under WQK's floors on zrckp, no read of the twelve is LOSS and at least one is GAIN; NOT_ADOPTED otherwise; INCOMPLETE when a comparison or the floors are missing, not WQK's or PSMU's, or not twelve reads (docs/PSMU_UNIFORM_LIFT_ON_T4.md, section 4; each arm its own verdict).

zrckpu (zrckp with each fixed column's lift above a uniform pool) against WQK's seed-0 zrckp fit of each split on the same T4, every read called under WQK's floors on zrckp: 0 GAIN (none), 1 LOSS (WQ-hotpotqa webqsp).

Next: with zrckpu1's verdict (PSMU's section 5): zrckpu1's next if it is adopted; if neither arm is adopted, the pool-softmax family closes for the MLP's input: the soft gap, the hard gap and the lift, each undivided and divided by 10, all NOT_ADOPTED under WQK's floors, so zrckp's input stays; the next forms under WQ go on, each in its own declared file: G6r under WQ on the T4 (WQK's section 9, item 1) and U2n's zrckpt refit on U2s's pick under WQ (WQP's section 7).

FC@5 and hit@1 are beside, deciding nothing.

| split | dataset | read | base R@5 | new R@5 | delta R@5 [95% CI] | filed | floor | call | delta FC@5 | delta hit@1 |
|---|---|---|---:|---:|---|---|---:|---|---:|---:|
| WQ-musique | metaqa | zero-shot | 0.4393 | 0.4172 | -0.0221 [-0.0270, -0.0171] | LOSS | 0.0621 | WITHIN | -0.0188 | -0.0165 |
| WQ-musique | squad | in-domain | 0.9106 | 0.9100 | -0.0005 [-0.0023, +0.0013] | WITHIN | 0.0075 | WITHIN | -0.0005 | -0.0011 |
| WQ-musique | musique | zero-shot | 0.4972 | 0.4829 | -0.0143 [-0.0200, -0.0083] | LOSS | 0.0566 | WITHIN | -0.0099 | -0.0004 |
| WQ-musique | hotpotqa | in-domain | 0.9051 | 0.9030 | -0.0022 [-0.0045, +0.0001] | WITHIN | 0.0087 | WITHIN | -0.0042 | +0.0046 |
| WQ-musique | 2wiki | in-domain | 0.9315 | 0.9287 | -0.0028 [-0.0044, -0.0012] | WITHIN | 0.0093 | WITHIN | -0.0097 | +0.0031 |
| WQ-musique | webqsp | in-domain | 0.6433 | 0.6391 | -0.0042 [-0.0134, +0.0047] | WITHIN | 0.0075 | WITHIN | -0.0073 | -0.0166 |
| WQ-hotpotqa | metaqa | zero-shot | 0.3927 | 0.4037 | +0.0111 [+0.0053, +0.0164] | GAIN | 0.0489 | WITHIN | +0.0079 | -0.0054 |
| WQ-hotpotqa | squad | in-domain | 0.9117 | 0.9132 | +0.0014 [-0.0006, +0.0035] | WITHIN | 0.0075 | WITHIN | +0.0014 | -0.0005 |
| WQ-hotpotqa | musique | in-domain | 0.5743 | 0.5742 | -0.0001 [-0.0049, +0.0048] | WITHIN | 0.0075 | WITHIN | +0.0008 | -0.0004 |
| WQ-hotpotqa | hotpotqa | zero-shot | 0.8525 | 0.8560 | +0.0035 [+0.0004, +0.0068] | WITHIN | 0.0089 | WITHIN | +0.0070 | -0.0120 |
| WQ-hotpotqa | 2wiki | in-domain | 0.9299 | 0.9278 | -0.0021 [-0.0038, -0.0005] | WITHIN | 0.0075 | WITHIN | -0.0064 | -0.0023 |
| WQ-hotpotqa | webqsp | in-domain | 0.6382 | 0.6281 | -0.0101 [-0.0184, -0.0020] | LOSS | 0.0075 | **LOSS** | -0.0140 | -0.0240 |
