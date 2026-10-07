# Screen L-2wiki (p@swa, s1eval carves)

## Against outputs\step1\fits\L-2wiki (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6468 | 0.6511 | +0.0043 [+0.0013, +0.0072] | WITHIN | 0.0047 | +0.0058 | +0.0040 |
| squad | in-domain | 11873 | 0.9112 | 0.9090 | -0.0022 [-0.0043, -0.0001] | WITHIN | 0.9054 | -0.0022 | -0.0028 |
| musique | in-domain | 2417 | 0.5601 | 0.5555 | -0.0047 [-0.0111, +0.0014] | WITHIN | 0.4734 | -0.0004 | -0.0054 |
| hotpotqa | in-domain | 7405 | 0.8977 | 0.9007 | +0.0030 [+0.0005, +0.0056] | WITHIN | 0.6846 | +0.0062 | -0.0082 |
| 2wiki | zero-shot | 12576 | 0.8074 | 0.8052 | -0.0021 [-0.0046, +0.0001] | WITHIN | 0.6079 | +0.0038 | -0.0487 |
| webqsp | zero-shot | 1503 | 0.1075 | 0.1202 | +0.0127 [+0.0054, +0.0207] | GAIN | 0.0535 | +0.0106 | +0.0120 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
