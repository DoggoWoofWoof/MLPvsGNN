# Screen L-squad (p@swa, s1eval carves)

## Against outputs\step1\fits\L-squad (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6535 | 0.7785 | +0.1251 [+0.1199, +0.1305] | GAIN | 0.0047 | +0.1266 | +0.2509 |
| squad | zero-shot | 11873 | 0.8797 | 0.8893 | +0.0096 [+0.0067, +0.0125] | GAIN | 0.9054 | +0.0096 | +0.0124 |
| musique | in-domain | 2417 | 0.5743 | 0.5670 | -0.0073 [-0.0133, -0.0017] | WITHIN | 0.4734 | -0.0108 | -0.0099 |
| hotpotqa | in-domain | 7405 | 0.9045 | 0.9007 | -0.0037 [-0.0063, -0.0014] | WITHIN | 0.6846 | -0.0057 | +0.0134 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.8695 | -0.0029 [-0.0047, -0.0011] | WITHIN | 0.6079 | -0.0086 | +0.0084 |
| webqsp | zero-shot | 1503 | 0.1302 | 0.2711 | +0.1409 [+0.1187, +0.1642] | GAIN | 0.0535 | +0.1111 | +0.0951 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
