# Screen scr-kg-base-zspk-s1-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zspk-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8037 | 0.8010 | -0.0027 [-0.0038, -0.0016] | WITHIN | 0.0047 | -0.0022 | -0.0119 |
| squad | in-domain | 11873 | 0.9139 | 0.9140 | +0.0001 [-0.0019, +0.0021] | WITHIN | 0.9054 | +0.0001 | +0.0035 |
| musique | in-domain | 2417 | 0.5754 | 0.5762 | +0.0008 [-0.0045, +0.0061] | WITHIN | 0.4734 | +0.0021 | +0.0083 |
| hotpotqa | zero-shot | 7405 | 0.8550 | 0.8503 | -0.0047 [-0.0079, -0.0013] | WITHIN | 0.6846 | -0.0074 | -0.0078 |
| 2wiki | in-domain | 12576 | 0.9265 | 0.9379 | +0.0114 [+0.0096, +0.0131] | GAIN | 0.6079 | +0.0377 | -0.0025 |
| webqsp | zero-shot | 1503 | 0.2640 | 0.2474 | -0.0166 [-0.0273, -0.0060] | LOSS | 0.0535 | -0.0133 | -0.0146 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
