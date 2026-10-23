# Screen scr-u2h-zrckpw (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zrckp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8008 | 0.8007 | -0.0001 [-0.0010, +0.0007] | WITHIN | 0.0047 | -0.0001 | +0.0025 |
| squad | in-domain | 11873 | 0.9121 | 0.9106 | -0.0015 [-0.0035, +0.0004] | WITHIN | 0.9054 | -0.0015 | +0.0026 |
| musique | zero-shot | 2417 | 0.5153 | 0.5131 | -0.0021 [-0.0070, +0.0027] | WITHIN | 0.4734 | -0.0066 | +0.0124 |
| hotpotqa | in-domain | 7405 | 0.9055 | 0.9053 | -0.0002 [-0.0026, +0.0021] | WITHIN | 0.6846 | +0.0000 | +0.0081 |
| 2wiki | in-domain | 12576 | 0.9345 | 0.9384 | +0.0039 [+0.0022, +0.0055] | WITHIN | 0.6079 | +0.0109 | -0.0014 |
| webqsp | zero-shot | 1503 | 0.3561 | 0.3558 | -0.0003 [-0.0088, +0.0083] | WITHIN | 0.0535 | +0.0020 | -0.0053 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
