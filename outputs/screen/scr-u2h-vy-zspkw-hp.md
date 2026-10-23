# Screen scr-u2h-zspkw-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2g-y-zspk-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8050 | 0.8053 | +0.0002 [-0.0006, +0.0011] | WITHIN | 0.0047 | -0.0002 | -0.0005 |
| squad | in-domain | 11873 | 0.9145 | 0.9126 | -0.0019 [-0.0041, +0.0002] | WITHIN | 0.9054 | -0.0019 | +0.0010 |
| musique | in-domain | 2417 | 0.5807 | 0.5782 | -0.0025 [-0.0078, +0.0027] | WITHIN | 0.4734 | -0.0062 | -0.0004 |
| hotpotqa | zero-shot | 7405 | 0.7966 | 0.8454 | +0.0488 [+0.0446, +0.0529] | GAIN | 0.6846 | +0.0910 | +0.0026 |
| 2wiki | in-domain | 12576 | 0.9212 | 0.9373 | +0.0162 [+0.0136, +0.0188] | GAIN | 0.6079 | +0.0388 | -0.0022 |
| webqsp | zero-shot | 1503 | 0.2674 | 0.2743 | +0.0069 [-0.0027, +0.0166] | WITHIN | 0.0535 | +0.0040 | +0.0027 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
