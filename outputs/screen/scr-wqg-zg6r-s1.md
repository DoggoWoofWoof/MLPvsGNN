# Screen scr-wqg-zg6r-s1 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g6rw-zg6r (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4105 | 0.4244 | +0.0139 [+0.0085, +0.0194] | GAIN | 0.0047 | +0.0109 | -0.0009 |
| squad | in-domain | 11873 | 0.9105 | 0.9108 | +0.0003 [-0.0015, +0.0022] | WITHIN | 0.9054 | +0.0003 | +0.0028 |
| musique | zero-shot | 2417 | 0.4106 | 0.4608 | +0.0502 [+0.0426, +0.0579] | GAIN | 0.4734 | +0.0356 | +0.0670 |
| hotpotqa | in-domain | 7405 | 0.9134 | 0.9110 | -0.0024 [-0.0045, -0.0003] | WITHIN | 0.6846 | -0.0057 | -0.0100 |
| 2wiki | in-domain | 12576 | 0.9369 | 0.9407 | +0.0038 [+0.0025, +0.0051] | WITHIN | 0.6079 | +0.0102 | -0.0001 |
| webqsp | in-domain | 1503 | 0.6501 | 0.6426 | -0.0074 [-0.0170, +0.0017] | WITHIN | 0.0535 | -0.0140 | -0.0080 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
