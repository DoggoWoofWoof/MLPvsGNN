# Screen J5 (p@swa, s1eval carves)

## Against outputs\full_zrct\fits\J5 (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7796 | 0.7992 | +0.0196 [+0.0166, +0.0223] | GAIN | 0.0047 | +0.0161 | +0.0113 |
| squad | in-domain | 11873 | 0.9107 | 0.9108 | +0.0001 [-0.0022, +0.0024] | WITHIN | 0.9054 | +0.0001 | -0.0003 |
| musique | in-domain | 2417 | 0.5669 | 0.5688 | +0.0019 [-0.0061, +0.0098] | WITHIN | 0.4734 | -0.0062 | +0.0381 |
| hotpotqa | in-domain | 7405 | 0.9049 | 0.9037 | -0.0012 [-0.0047, +0.0022] | WITHIN | 0.6846 | -0.0043 | -0.0004 |
| 2wiki | in-domain | 12576 | 0.8697 | 0.9269 | +0.0573 [+0.0539, +0.0608] | GAIN | 0.6079 | +0.1095 | +0.0823 |
| webqsp | zero-shot | 1503 | 0.3071 | 0.3600 | +0.0529 [+0.0365, +0.0702] | GAIN | 0.0535 | +0.0499 | +0.0293 |

## Against outputs\step1\fits\J5 (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6443 | 0.7992 | +0.1549 [+0.1493, +0.1606] | GAIN | 0.0047 | +0.1573 | +0.2692 |
| squad | in-domain | 11873 | 0.9095 | 0.9108 | +0.0013 [-0.0009, +0.0035] | WITHIN | 0.9054 | +0.0013 | -0.0012 |
| musique | in-domain | 2417 | 0.5609 | 0.5688 | +0.0079 [-0.0004, +0.0154] | WITHIN | 0.4734 | +0.0070 | +0.0439 |
| hotpotqa | in-domain | 7405 | 0.9013 | 0.9037 | +0.0024 [-0.0010, +0.0059] | WITHIN | 0.6846 | +0.0028 | -0.0001 |
| 2wiki | in-domain | 12576 | 0.8694 | 0.9269 | +0.0575 [+0.0543, +0.0609] | GAIN | 0.6079 | +0.1070 | +0.0701 |
| webqsp | zero-shot | 1503 | 0.1167 | 0.3600 | +0.2432 [+0.2187, +0.2690] | GAIN | 0.0535 | +0.1989 | +0.0845 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
