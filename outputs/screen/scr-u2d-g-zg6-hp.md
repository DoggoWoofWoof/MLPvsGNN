# Screen scr-u2d-new-zg6-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zg6-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8112 | 0.8134 | +0.0022 [+0.0010, +0.0034] | WITHIN | 0.0047 | +0.0037 | +0.0045 |
| squad | in-domain | 11873 | 0.9118 | 0.9137 | +0.0019 [-0.0004, +0.0040] | WITHIN | 0.9054 | +0.0019 | +0.0029 |
| musique | in-domain | 2417 | 0.5937 | 0.5985 | +0.0048 [-0.0008, +0.0105] | WITHIN | 0.4734 | +0.0033 | -0.0074 |
| hotpotqa | zero-shot | 7405 | 0.8309 | 0.7810 | -0.0499 [-0.0545, -0.0450] | LOSS | 0.6846 | -0.1011 | -0.0020 |
| 2wiki | in-domain | 12576 | 0.9391 | 0.9267 | -0.0124 [-0.0152, -0.0098] | LOSS | 0.6079 | -0.0305 | +0.0035 |
| webqsp | zero-shot | 1503 | 0.2189 | 0.1781 | -0.0408 [-0.0566, -0.0243] | LOSS | 0.0535 | -0.0339 | -0.0319 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
