# Screen L-squad (p@swa, s1eval carves)

## Against outputs\step1\fits\L-squad (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6535 | 0.7256 | +0.0721 [+0.0678, +0.0767] | GAIN | 0.0047 | +0.0626 | +0.1421 |
| squad | zero-shot | 11873 | 0.8797 | 0.8811 | +0.0013 [-0.0014, +0.0040] | WITHIN | 0.9054 | +0.0013 | -0.0032 |
| musique | in-domain | 2417 | 0.5743 | 0.5650 | -0.0093 [-0.0150, -0.0032] | LOSS | 0.4734 | -0.0141 | -0.0054 |
| hotpotqa | in-domain | 7405 | 0.9045 | 0.9034 | -0.0010 [-0.0034, +0.0015] | WITHIN | 0.6846 | -0.0008 | +0.0045 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.8745 | +0.0022 [+0.0002, +0.0041] | WITHIN | 0.6079 | +0.0049 | +0.0003 |
| webqsp | zero-shot | 1503 | 0.1302 | 0.2429 | +0.1127 [+0.0942, +0.1313] | GAIN | 0.0535 | +0.0898 | +0.0605 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
