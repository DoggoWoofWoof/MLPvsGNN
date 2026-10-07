# Screen scr-vshare (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.6508 | -0.0030 [-0.0056, -0.0002] | WITHIN | 0.0047 | -0.0025 | +0.0031 |
| squad | in-domain | 11873 | 0.9100 | 0.9095 | -0.0004 [-0.0024, +0.0015] | WITHIN | 0.9054 | -0.0004 | +0.0012 |
| musique | zero-shot | 2417 | 0.2696 | 0.3586 | +0.0891 [+0.0793, +0.0990] | GAIN | 0.4734 | +0.0430 | +0.1489 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.8998 | -0.0017 [-0.0042, +0.0005] | WITHIN | 0.6846 | -0.0039 | -0.0007 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8722 | -0.0011 [-0.0030, +0.0010] | WITHIN | 0.6079 | -0.0024 | -0.0068 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.1824 | +0.0045 [-0.0055, +0.0155] | WITHIN | 0.0535 | +0.0080 | -0.0013 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
