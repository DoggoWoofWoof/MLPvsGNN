# Screen scr-gnx-nx-s0-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-ml1-zg6r-s0-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3519 | 0.3629 | +0.0110 [+0.0056, +0.0167] | GAIN | 0.0047 | +0.0068 | +0.0181 |
| squad | in-domain | 11873 | 0.9139 | 0.9148 | +0.0009 [-0.0011, +0.0029] | WITHIN | 0.9054 | +0.0009 | +0.0030 |
| musique | in-domain | 2417 | 0.5912 | 0.5914 | +0.0003 [-0.0054, +0.0060] | WITHIN | 0.4734 | -0.0008 | -0.0091 |
| hotpotqa | zero-shot | 7405 | 0.8004 | 0.8008 | +0.0004 [-0.0031, +0.0039] | WITHIN | 0.6846 | +0.0005 | -0.0045 |
| 2wiki | in-domain | 12576 | 0.9296 | 0.9270 | -0.0026 [-0.0044, -0.0010] | WITHIN | 0.6079 | -0.0062 | -0.0017 |
| webqsp | in-domain | 1503 | 0.6658 | 0.6636 | -0.0023 [-0.0117, +0.0077] | WITHIN | 0.0535 | +0.0033 | -0.0067 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
