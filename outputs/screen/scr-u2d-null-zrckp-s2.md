# Screen scr-u2d-base-zrckp-s2 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zrckp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8008 | 0.8043 | +0.0034 [+0.0023, +0.0046] | WITHIN | 0.0047 | +0.0036 | +0.0114 |
| squad | in-domain | 11873 | 0.9121 | 0.9111 | -0.0009 [-0.0028, +0.0011] | WITHIN | 0.9054 | -0.0009 | +0.0044 |
| musique | zero-shot | 2417 | 0.5153 | 0.5157 | +0.0004 [-0.0048, +0.0053] | WITHIN | 0.4734 | -0.0029 | -0.0029 |
| hotpotqa | in-domain | 7405 | 0.9055 | 0.9066 | +0.0011 [-0.0011, +0.0033] | WITHIN | 0.6846 | +0.0039 | -0.0004 |
| 2wiki | in-domain | 12576 | 0.9345 | 0.9362 | +0.0017 [+0.0001, +0.0032] | WITHIN | 0.6079 | +0.0061 | -0.0025 |
| webqsp | zero-shot | 1503 | 0.3561 | 0.3686 | +0.0124 [+0.0033, +0.0220] | GAIN | 0.0535 | +0.0126 | -0.0007 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
