# Screen scr-zrct-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrm-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7801 | 0.7793 | -0.0008 [-0.0017, +0.0001] | WITHIN | 0.0047 | -0.0004 | -0.0003 |
| squad | in-domain | 11873 | 0.9116 | 0.9112 | -0.0003 [-0.0023, +0.0014] | WITHIN | 0.9054 | -0.0003 | +0.0011 |
| musique | in-domain | 2417 | 0.5600 | 0.5645 | +0.0045 [-0.0006, +0.0098] | WITHIN | 0.4734 | -0.0021 | -0.0120 |
| hotpotqa | zero-shot | 7405 | 0.8457 | 0.8451 | -0.0006 [-0.0035, +0.0023] | WITHIN | 0.6846 | -0.0008 | -0.0043 |
| 2wiki | in-domain | 12576 | 0.8681 | 0.8731 | +0.0051 [+0.0033, +0.0069] | WITHIN | 0.6079 | +0.0146 | -0.0024 |
| webqsp | zero-shot | 1503 | 0.2416 | 0.2865 | +0.0450 [+0.0310, +0.0586] | GAIN | 0.0535 | +0.0432 | -0.0160 |

## Against outputs\full_zret\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6419 | 0.7793 | +0.1374 [+0.1319, +0.1430] | GAIN | 0.0047 | +0.1435 | +0.2648 |
| squad | in-domain | 11873 | 0.9127 | 0.9112 | -0.0015 [-0.0035, +0.0006] | WITHIN | 0.9054 | -0.0015 | +0.0018 |
| musique | in-domain | 2417 | 0.5623 | 0.5645 | +0.0022 [-0.0043, +0.0084] | WITHIN | 0.4734 | -0.0017 | -0.0124 |
| hotpotqa | zero-shot | 7405 | 0.8445 | 0.8451 | +0.0006 [-0.0029, +0.0043] | WITHIN | 0.6846 | +0.0016 | -0.0005 |
| 2wiki | in-domain | 12576 | 0.8728 | 0.8731 | +0.0003 [-0.0017, +0.0024] | WITHIN | 0.6079 | +0.0015 | -0.0053 |
| webqsp | zero-shot | 1503 | 0.1195 | 0.2865 | +0.1670 [+0.1454, +0.1885] | GAIN | 0.0535 | +0.1384 | +0.0519 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7793 | +0.1253 [+0.1200, +0.1305] | GAIN | 0.0047 | +0.1335 | +0.2553 |
| squad | in-domain | 11873 | 0.9116 | 0.9112 | -0.0004 [-0.0028, +0.0021] | WITHIN | 0.9054 | -0.0004 | +0.0031 |
| musique | in-domain | 2417 | 0.5550 | 0.5645 | +0.0094 [+0.0032, +0.0160] | GAIN | 0.4734 | +0.0157 | +0.0004 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8451 | -0.0134 [-0.0172, -0.0095] | LOSS | 0.6846 | -0.0243 | +0.0000 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8731 | +0.0059 [+0.0036, +0.0084] | WITHIN | 0.6079 | +0.0173 | -0.0088 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2865 | +0.1666 [+0.1440, +0.1882] | GAIN | 0.0535 | +0.1364 | +0.0472 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
