# Screen scr-a13x-zspxb-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g5-zspx-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8039 | 0.8036 | -0.0003 [-0.0011, +0.0004] | WITHIN | 0.0047 | -0.0010 | -0.0041 |
| squad | in-domain | 11873 | 0.9129 | 0.9152 | +0.0023 [+0.0006, +0.0040] | WITHIN | 0.9054 | +0.0023 | +0.0004 |
| musique | in-domain | 2417 | 0.5901 | 0.5836 | -0.0065 [-0.0111, -0.0017] | WITHIN | 0.4734 | -0.0128 | -0.0041 |
| hotpotqa | zero-shot | 7405 | 0.8547 | 0.8560 | +0.0013 [-0.0013, +0.0040] | WITHIN | 0.6846 | +0.0011 | +0.0080 |
| 2wiki | in-domain | 12576 | 0.9312 | 0.9297 | -0.0015 [-0.0030, -0.0000] | WITHIN | 0.6079 | -0.0078 | -0.0060 |
| webqsp | zero-shot | 1503 | 0.2914 | 0.2781 | -0.0133 [-0.0216, -0.0055] | LOSS | 0.0535 | -0.0080 | +0.0060 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8036 | +0.1497 [+0.1440, +0.1550] | GAIN | 0.0047 | +0.1538 | +0.2884 |
| squad | in-domain | 11873 | 0.9116 | 0.9152 | +0.0035 [+0.0013, +0.0059] | WITHIN | 0.9054 | +0.0035 | +0.0023 |
| musique | in-domain | 2417 | 0.5550 | 0.5836 | +0.0286 [+0.0210, +0.0360] | GAIN | 0.4734 | +0.0294 | +0.0509 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8560 | -0.0026 [-0.0074, +0.0021] | WITHIN | 0.6846 | -0.0055 | -0.0158 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9297 | +0.0625 [+0.0591, +0.0662] | GAIN | 0.6079 | +0.1233 | +0.0687 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2781 | +0.1581 [+0.1342, +0.1819] | GAIN | 0.0535 | +0.1191 | +0.0971 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
