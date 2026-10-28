# Screen scr-psmh-zrckph-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3927 | 0.4282 | +0.0355 [+0.0296, +0.0411] | GAIN | 0.0047 | +0.0289 | +0.0017 |
| squad | in-domain | 11873 | 0.9117 | 0.9128 | +0.0011 [-0.0010, +0.0033] | WITHIN | 0.9054 | +0.0011 | -0.0001 |
| musique | in-domain | 2417 | 0.5743 | 0.5760 | +0.0017 [-0.0030, +0.0065] | WITHIN | 0.4734 | +0.0033 | +0.0012 |
| hotpotqa | zero-shot | 7405 | 0.8525 | 0.8469 | -0.0056 [-0.0090, -0.0022] | WITHIN | 0.6846 | -0.0070 | -0.0192 |
| 2wiki | in-domain | 12576 | 0.9299 | 0.9264 | -0.0036 [-0.0052, -0.0018] | WITHIN | 0.6079 | -0.0088 | -0.0017 |
| webqsp | in-domain | 1503 | 0.6382 | 0.6243 | -0.0139 [-0.0225, -0.0054] | LOSS | 0.0535 | -0.0193 | -0.0126 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
