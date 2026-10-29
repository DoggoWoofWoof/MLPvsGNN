# Screen scr-wqg-zg6-s1-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zg6-s0-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4167 | 0.4129 | -0.0037 [-0.0086, +0.0012] | WITHIN | 0.0047 | -0.0088 | +0.0427 |
| squad | in-domain | 11873 | 0.9114 | 0.9110 | -0.0004 [-0.0024, +0.0016] | WITHIN | 0.9054 | -0.0004 | +0.0007 |
| musique | in-domain | 2417 | 0.5961 | 0.5896 | -0.0065 [-0.0114, -0.0015] | WITHIN | 0.4734 | -0.0062 | -0.0041 |
| hotpotqa | zero-shot | 7405 | 0.8183 | 0.8261 | +0.0078 [+0.0046, +0.0113] | GAIN | 0.6846 | +0.0120 | -0.0211 |
| 2wiki | in-domain | 12576 | 0.9414 | 0.9397 | -0.0017 [-0.0031, -0.0003] | WITHIN | 0.6079 | -0.0021 | -0.0052 |
| webqsp | in-domain | 1503 | 0.6421 | 0.6340 | -0.0081 [-0.0197, +0.0034] | WITHIN | 0.0535 | -0.0080 | -0.0020 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
