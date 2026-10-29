# Screen scr-wqg-zg6r-s2-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g6rw-zg6r-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3834 | 0.4156 | +0.0322 [+0.0270, +0.0373] | GAIN | 0.0047 | +0.0272 | -0.0086 |
| squad | in-domain | 11873 | 0.9111 | 0.9124 | +0.0013 [-0.0004, +0.0031] | WITHIN | 0.9054 | +0.0013 | -0.0012 |
| musique | in-domain | 2417 | 0.5941 | 0.5977 | +0.0037 [-0.0013, +0.0089] | WITHIN | 0.4734 | +0.0087 | -0.0021 |
| hotpotqa | zero-shot | 7405 | 0.8529 | 0.8485 | -0.0044 [-0.0073, -0.0016] | WITHIN | 0.6846 | -0.0057 | +0.0057 |
| 2wiki | in-domain | 12576 | 0.9420 | 0.9432 | +0.0012 [-0.0002, +0.0026] | WITHIN | 0.6079 | +0.0050 | -0.0043 |
| webqsp | in-domain | 1503 | 0.6536 | 0.6401 | -0.0135 [-0.0239, -0.0037] | LOSS | 0.0535 | -0.0146 | -0.0173 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
