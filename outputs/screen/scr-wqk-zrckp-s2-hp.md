# Screen scr-wqk-zrckp-s2-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3927 | 0.4147 | +0.0220 [+0.0159, +0.0279] | GAIN | 0.0047 | +0.0173 | +0.0053 |
| squad | in-domain | 11873 | 0.9117 | 0.9116 | -0.0001 [-0.0022, +0.0020] | WITHIN | 0.9054 | -0.0001 | +0.0003 |
| musique | in-domain | 2417 | 0.5743 | 0.5786 | +0.0043 [-0.0009, +0.0090] | WITHIN | 0.4734 | +0.0095 | +0.0120 |
| hotpotqa | zero-shot | 7405 | 0.8525 | 0.8583 | +0.0059 [+0.0028, +0.0089] | WITHIN | 0.6846 | +0.0119 | -0.0046 |
| 2wiki | in-domain | 12576 | 0.9299 | 0.9313 | +0.0013 [-0.0003, +0.0030] | WITHIN | 0.6079 | +0.0029 | -0.0083 |
| webqsp | in-domain | 1503 | 0.6382 | 0.6393 | +0.0011 [-0.0080, +0.0105] | WITHIN | 0.0535 | -0.0060 | +0.0126 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
