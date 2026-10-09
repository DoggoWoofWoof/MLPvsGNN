# Screen scr-zrct-hp (p@swa, s1eval carves)

## Against outputs\step1\fits\L-hotpotqa (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7793 | +0.1253 [+0.1200, +0.1305] | GAIN | 0.0047 | +0.1335 | +0.2553 |
| squad | in-domain | 11873 | 0.9116 | 0.9112 | -0.0004 [-0.0028, +0.0021] | WITHIN | 0.9054 | -0.0004 | +0.0031 |
| musique | in-domain | 2417 | 0.5550 | 0.5645 | +0.0094 [+0.0032, +0.0160] | GAIN | 0.4734 | +0.0157 | +0.0004 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8451 | -0.0134 [-0.0172, -0.0095] | LOSS | 0.6846 | -0.0243 | +0.0000 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8731 | +0.0059 [+0.0036, +0.0084] | WITHIN | 0.6079 | +0.0173 | -0.0088 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2865 | +0.1666 [+0.1440, +0.1882] | GAIN | 0.0535 | +0.1364 | +0.0472 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
