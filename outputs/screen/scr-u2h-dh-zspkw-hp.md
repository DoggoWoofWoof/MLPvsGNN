# Screen scr-u2h-zspkw-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zspk-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8043 | 0.8053 | +0.0009 [+0.0001, +0.0018] | WITHIN | 0.0047 | +0.0008 | +0.0008 |
| squad | in-domain | 11873 | 0.9117 | 0.9126 | +0.0008 [-0.0011, +0.0028] | WITHIN | 0.9054 | +0.0008 | +0.0016 |
| musique | in-domain | 2417 | 0.5800 | 0.5782 | -0.0018 [-0.0064, +0.0033] | WITHIN | 0.4734 | -0.0066 | +0.0145 |
| hotpotqa | zero-shot | 7405 | 0.8512 | 0.8454 | -0.0059 [-0.0092, -0.0025] | WITHIN | 0.6846 | -0.0124 | -0.0042 |
| 2wiki | in-domain | 12576 | 0.9290 | 0.9373 | +0.0083 [+0.0066, +0.0102] | GAIN | 0.6079 | +0.0200 | +0.0062 |
| webqsp | zero-shot | 1503 | 0.2773 | 0.2743 | -0.0030 [-0.0115, +0.0051] | WITHIN | 0.0535 | -0.0033 | +0.0000 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
