# C1R: Claim 1 under WQ on PKO's pools, zrckp against zg6r

**CLAIM1_NOT_ALL** (bar 0.9, 2000 resamples, seed-0 fits)

| split | dataset | read | MLP | GNN | RRF | GNN gain [lo, hi] | share [lo, hi] | clause |
|---|---|---|---:|---:|---:|---|---|---|
| WQ-musique | metaqa | zero-shot | 0.3992 | 0.3615 | 0.0047 | +0.3568 [+0.3478, +0.3658] | 1.106 [1.085, 1.127] | HOLDS |
| WQ-musique | squad | in-domain | 0.9139 | 0.9142 | 0.9054 | +0.0088 [+0.0055, +0.0120] | 0.971 [0.761, 1.239] | NOT_SHOWN |
| WQ-musique | musique | zero-shot | 0.5051 | 0.5086 | 0.4734 | +0.0352 [+0.0244, +0.0455] | 0.901 [0.727, 1.097] | HOLDS |
| WQ-musique | hotpotqa | in-domain | 0.8423 | 0.8593 | 0.6846 | +0.1747 [+0.1682, +0.1809] | 0.903 [0.885, 0.920] | NOT_SHOWN |
| WQ-musique | 2wiki | in-domain | 0.9045 | 0.9269 | 0.6079 | +0.3190 [+0.3141, +0.3236] | 0.930 [0.923, 0.937] | HOLDS |
| WQ-musique | webqsp | in-domain | 0.6562 | 0.6632 | 0.0535 | +0.6097 [+0.5849, +0.6314] | 0.989 [0.965, 1.013] | HOLDS |
| WQ-hotpotqa | metaqa | zero-shot | 0.3949 | 0.3519 | 0.0047 | +0.3472 [+0.3383, +0.3556] | 1.124 [1.101, 1.147] | HOLDS |
| WQ-hotpotqa | squad | in-domain | 0.9138 | 0.9139 | 0.9054 | +0.0085 [+0.0051, +0.0120] | 0.990 [0.735, 1.309] | NOT_SHOWN |
| WQ-hotpotqa | musique | in-domain | 0.5777 | 0.5912 | 0.4734 | +0.1177 [+0.1076, +0.1274] | 0.886 [0.832, 0.940] | MISSES |
| WQ-hotpotqa | hotpotqa | zero-shot | 0.8110 | 0.8004 | 0.6846 | +0.1158 [+0.1090, +0.1223] | 1.091 [1.054, 1.132] | HOLDS |
| WQ-hotpotqa | 2wiki | in-domain | 0.9029 | 0.9296 | 0.6079 | +0.3217 [+0.3169, +0.3264] | 0.917 [0.910, 0.924] | HOLDS |
| WQ-hotpotqa | webqsp | in-domain | 0.6569 | 0.6658 | 0.0535 | +0.6123 [+0.5885, +0.6352] | 0.985 [0.962, 1.008] | HOLDS |

Zero-shot clauses (R@5): excess = zero-shot gap minus the allowed in-domain gap (floored at 0).

| read | in-domain split | zs gap | allowed | excess [lo, hi] | call |
|---|---|---:|---:|---|---|
| WQ-musique musique | WQ-hotpotqa | +0.0035 | 0.0134 | -0.0099 [-0.0193, -0.0009] | HOLDS |
| WQ-hotpotqa hotpotqa | WQ-musique | -0.0106 | 0.0170 | -0.0276 [-0.0328, -0.0224] | HOLDS |
| WQ-musique metaqa | none (allowed 0) | -0.0377 | 0.0000 | -0.0377 [-0.0449, -0.0308] | HOLDS |
| WQ-hotpotqa metaqa | none (allowed 0) | -0.0430 | 0.0000 | -0.0430 [-0.0505, -0.0356] | HOLDS |
