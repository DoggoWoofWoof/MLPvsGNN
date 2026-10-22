# Screen scr-u2g-y-zspk-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zspk-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8043 | 0.8050 | +0.0007 [-0.0003, +0.0017] | WITHIN | 0.0047 | +0.0010 | +0.0013 |
| squad | in-domain | 11873 | 0.9117 | 0.9145 | +0.0028 [+0.0005, +0.0051] | WITHIN | 0.9054 | +0.0028 | +0.0006 |
| musique | in-domain | 2417 | 0.5800 | 0.5807 | +0.0007 [-0.0049, +0.0068] | WITHIN | 0.4734 | -0.0004 | +0.0149 |
| hotpotqa | zero-shot | 7405 | 0.8512 | 0.7966 | -0.0546 [-0.0592, -0.0502] | LOSS | 0.6846 | -0.1034 | -0.0068 |
| 2wiki | in-domain | 12576 | 0.9290 | 0.9212 | -0.0079 [-0.0106, -0.0048] | LOSS | 0.6079 | -0.0188 | +0.0084 |
| webqsp | zero-shot | 1503 | 0.2773 | 0.2674 | -0.0099 [-0.0191, -0.0004] | LOSS | 0.0535 | -0.0073 | -0.0027 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
