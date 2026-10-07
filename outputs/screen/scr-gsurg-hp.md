# Screen scr-gsurg-hp (p@swa, s1eval carves)

## Against outputs\step1\fits\L-hotpotqa (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.6556 | +0.0016 [-0.0011, +0.0043] | WITHIN | 0.0047 | +0.0035 | +0.0088 |
| squad | in-domain | 11873 | 0.9116 | 0.9097 | -0.0019 [-0.0043, +0.0004] | WITHIN | 0.9054 | -0.0019 | -0.0034 |
| musique | in-domain | 2417 | 0.5550 | 0.5599 | +0.0049 [-0.0006, +0.0106] | WITHIN | 0.4734 | +0.0174 | -0.0041 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8438 | -0.0147 [-0.0188, -0.0108] | LOSS | 0.6846 | -0.0280 | -0.0073 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8763 | +0.0091 [+0.0069, +0.0114] | GAIN | 0.6079 | +0.0257 | -0.0107 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.1324 | +0.0124 [+0.0020, +0.0227] | GAIN | 0.0535 | +0.0093 | -0.0040 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
