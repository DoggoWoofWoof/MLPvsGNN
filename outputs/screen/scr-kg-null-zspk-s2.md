# Screen scr-kg-base-zspk-s2 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zspk (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8013 | 0.8044 | +0.0031 [+0.0020, +0.0042] | WITHIN | 0.0047 | +0.0039 | +0.0039 |
| squad | in-domain | 11873 | 0.9102 | 0.9114 | +0.0012 [-0.0008, +0.0030] | WITHIN | 0.9054 | +0.0012 | +0.0029 |
| musique | zero-shot | 2417 | 0.5146 | 0.5074 | -0.0072 [-0.0129, -0.0016] | WITHIN | 0.4734 | -0.0033 | -0.0008 |
| hotpotqa | in-domain | 7405 | 0.9086 | 0.9068 | -0.0018 [-0.0039, +0.0004] | WITHIN | 0.6846 | -0.0051 | +0.0161 |
| 2wiki | in-domain | 12576 | 0.9364 | 0.9353 | -0.0011 [-0.0025, +0.0003] | WITHIN | 0.6079 | -0.0017 | -0.0005 |
| webqsp | zero-shot | 1503 | 0.3151 | 0.3158 | +0.0007 [-0.0073, +0.0087] | WITHIN | 0.0535 | +0.0007 | -0.0060 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
