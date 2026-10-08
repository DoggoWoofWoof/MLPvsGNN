# Screen J5 (p@swa, s1eval carves)

## Against outputs\step1\fits\J5 (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6443 | 0.7636 | +0.1193 [+0.1145, +0.1244] | GAIN | 0.0047 | +0.1203 | +0.1914 |
| squad | in-domain | 11873 | 0.9095 | 0.9109 | +0.0013 [-0.0010, +0.0036] | WITHIN | 0.9054 | +0.0013 | -0.0019 |
| musique | in-domain | 2417 | 0.5609 | 0.5640 | +0.0031 [-0.0033, +0.0095] | WITHIN | 0.4734 | +0.0112 | +0.0037 |
| hotpotqa | in-domain | 7405 | 0.9013 | 0.9008 | -0.0005 [-0.0034, +0.0024] | WITHIN | 0.6846 | -0.0014 | +0.0049 |
| 2wiki | in-domain | 12576 | 0.8694 | 0.8659 | -0.0035 [-0.0055, -0.0012] | WITHIN | 0.6079 | -0.0072 | +0.0009 |
| webqsp | zero-shot | 1503 | 0.1167 | 0.2329 | +0.1162 [+0.0969, +0.1353] | GAIN | 0.0535 | +0.0925 | +0.0559 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
