# Screen J5 (p@swa, s1eval carves)

## Against outputs\step1\fits\J5 (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6443 | 0.7796 | +0.1353 [+0.1299, +0.1409] | GAIN | 0.0047 | +0.1411 | +0.2578 |
| squad | in-domain | 11873 | 0.9095 | 0.9107 | +0.0012 [-0.0011, +0.0035] | WITHIN | 0.9054 | +0.0012 | -0.0009 |
| musique | in-domain | 2417 | 0.5609 | 0.5669 | +0.0060 [-0.0001, +0.0120] | WITHIN | 0.4734 | +0.0132 | +0.0058 |
| hotpotqa | in-domain | 7405 | 0.9013 | 0.9049 | +0.0036 [+0.0009, +0.0063] | WITHIN | 0.6846 | +0.0072 | +0.0003 |
| 2wiki | in-domain | 12576 | 0.8694 | 0.8697 | +0.0003 [-0.0018, +0.0024] | WITHIN | 0.6079 | -0.0025 | -0.0122 |
| webqsp | zero-shot | 1503 | 0.1167 | 0.3071 | +0.1904 [+0.1683, +0.2133] | GAIN | 0.0535 | +0.1490 | +0.0552 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
