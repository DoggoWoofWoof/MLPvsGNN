# Screen scr-wqk-zspk-s1-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zspk-s0-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3384 | 0.3599 | +0.0215 [+0.0159, +0.0270] | GAIN | 0.0047 | +0.0189 | -0.0362 |
| squad | in-domain | 11873 | 0.9114 | 0.9107 | -0.0007 [-0.0027, +0.0013] | WITHIN | 0.9054 | -0.0007 | +0.0029 |
| musique | in-domain | 2417 | 0.5794 | 0.5766 | -0.0028 [-0.0079, +0.0024] | WITHIN | 0.4734 | -0.0033 | -0.0017 |
| hotpotqa | zero-shot | 7405 | 0.8496 | 0.8484 | -0.0012 [-0.0044, +0.0020] | WITHIN | 0.6846 | +0.0004 | -0.0352 |
| 2wiki | in-domain | 12576 | 0.9330 | 0.9345 | +0.0015 [-0.0002, +0.0030] | WITHIN | 0.6079 | +0.0046 | +0.0012 |
| webqsp | in-domain | 1503 | 0.6153 | 0.6174 | +0.0020 [-0.0077, +0.0127] | WITHIN | 0.0535 | +0.0040 | -0.0160 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
