# Screen J5 (p@swa, s1eval carves)

## Against outputs\step1\fits\J5 (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6443 | 0.7202 | +0.0759 [+0.0715, +0.0805] | GAIN | 0.0047 | +0.0706 | +0.1360 |
| squad | in-domain | 11873 | 0.9095 | 0.9083 | -0.0013 [-0.0032, +0.0007] | WITHIN | 0.9054 | -0.0013 | -0.0024 |
| musique | in-domain | 2417 | 0.5609 | 0.5595 | -0.0013 [-0.0071, +0.0049] | WITHIN | 0.4734 | +0.0037 | -0.0021 |
| hotpotqa | in-domain | 7405 | 0.9013 | 0.9032 | +0.0019 [-0.0007, +0.0045] | WITHIN | 0.6846 | +0.0038 | -0.0090 |
| 2wiki | in-domain | 12576 | 0.8694 | 0.8696 | +0.0001 [-0.0016, +0.0018] | WITHIN | 0.6079 | -0.0011 | -0.0064 |
| webqsp | zero-shot | 1503 | 0.1167 | 0.2181 | +0.1013 [+0.0841, +0.1190] | GAIN | 0.0535 | +0.0772 | +0.0605 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
