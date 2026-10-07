# Screen J5 (p@swa, s1eval carves)

## Against outputs\step1\fits\J5 (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6443 | 0.6461 | +0.0018 [-0.0013, +0.0050] | WITHIN | 0.0047 | +0.0050 | -0.0123 |
| squad | in-domain | 11873 | 0.9095 | 0.9086 | -0.0009 [-0.0032, +0.0013] | WITHIN | 0.9054 | -0.0009 | -0.0030 |
| musique | in-domain | 2417 | 0.5609 | 0.5526 | -0.0082 [-0.0140, -0.0024] | LOSS | 0.4734 | -0.0058 | -0.0095 |
| hotpotqa | in-domain | 7405 | 0.9013 | 0.8984 | -0.0029 [-0.0057, -0.0002] | WITHIN | 0.6846 | -0.0035 | -0.0115 |
| 2wiki | in-domain | 12576 | 0.8694 | 0.8667 | -0.0028 [-0.0048, -0.0008] | WITHIN | 0.6079 | -0.0056 | -0.0139 |
| webqsp | zero-shot | 1503 | 0.1167 | 0.1079 | -0.0089 [-0.0177, +0.0005] | WITHIN | 0.0535 | -0.0080 | +0.0027 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
