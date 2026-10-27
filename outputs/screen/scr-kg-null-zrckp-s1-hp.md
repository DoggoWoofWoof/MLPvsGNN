# Screen scr-kg-base-zrckp-s1-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zrckp-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8038 | 0.8013 | -0.0024 [-0.0036, -0.0014] | WITHIN | 0.0047 | -0.0020 | -0.0097 |
| squad | in-domain | 11873 | 0.9117 | 0.9108 | -0.0009 [-0.0031, +0.0012] | WITHIN | 0.9054 | -0.0009 | +0.0025 |
| musique | in-domain | 2417 | 0.5743 | 0.5828 | +0.0085 [+0.0029, +0.0139] | GAIN | 0.4734 | +0.0120 | +0.0124 |
| hotpotqa | zero-shot | 7405 | 0.8504 | 0.8562 | +0.0058 [+0.0030, +0.0088] | WITHIN | 0.6846 | +0.0117 | +0.0103 |
| 2wiki | in-domain | 12576 | 0.9354 | 0.9352 | -0.0002 [-0.0018, +0.0012] | WITHIN | 0.6079 | -0.0005 | -0.0016 |
| webqsp | zero-shot | 1503 | 0.3289 | 0.3415 | +0.0126 [+0.0012, +0.0242] | GAIN | 0.0535 | +0.0086 | -0.0053 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
