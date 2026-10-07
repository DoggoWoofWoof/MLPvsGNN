# Screen scr-gcs-hp (p@swa, s1eval carves)

## Against outputs\step1\fits\L-hotpotqa (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.6455 | -0.0084 [-0.0117, -0.0054] | LOSS | 0.0047 | -0.0064 | -0.0020 |
| squad | in-domain | 11873 | 0.9116 | 0.9107 | -0.0009 [-0.0030, +0.0013] | WITHIN | 0.9054 | -0.0009 | +0.0035 |
| musique | in-domain | 2417 | 0.5550 | 0.5583 | +0.0032 [-0.0021, +0.0083] | WITHIN | 0.4734 | +0.0103 | +0.0025 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8453 | -0.0132 [-0.0169, -0.0095] | LOSS | 0.6846 | -0.0259 | -0.0041 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8722 | +0.0050 [+0.0030, +0.0072] | WITHIN | 0.6079 | +0.0160 | -0.0037 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.1194 | -0.0005 [-0.0095, +0.0083] | WITHIN | 0.0535 | -0.0013 | +0.0013 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
