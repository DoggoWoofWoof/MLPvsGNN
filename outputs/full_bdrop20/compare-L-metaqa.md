# Screen L-metaqa (p@swa, s1eval carves)

## Against outputs\step1\fits\L-metaqa (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.0773 | 0.0724 | -0.0049 [-0.0076, -0.0021] | WITHIN | 0.0047 | -0.0050 | -0.0009 |
| squad | in-domain | 11873 | 0.9076 | 0.9095 | +0.0019 [-0.0002, +0.0040] | WITHIN | 0.9054 | +0.0019 | +0.0040 |
| musique | in-domain | 2417 | 0.5614 | 0.5583 | -0.0031 [-0.0089, +0.0029] | WITHIN | 0.4734 | -0.0066 | +0.0004 |
| hotpotqa | in-domain | 7405 | 0.9046 | 0.8995 | -0.0051 [-0.0076, -0.0026] | WITHIN | 0.6846 | -0.0082 | -0.0142 |
| 2wiki | in-domain | 12576 | 0.8639 | 0.8661 | +0.0021 [+0.0001, +0.0042] | WITHIN | 0.6079 | +0.0048 | -0.0074 |
| webqsp | zero-shot | 1503 | 0.0926 | 0.0845 | -0.0081 [-0.0161, -0.0003] | LOSS | 0.0535 | -0.0060 | -0.0020 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
