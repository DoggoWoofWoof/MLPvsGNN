# Screen scr-a11x-zspx4 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g5-zspx (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8030 | 0.8018 | -0.0012 [-0.0020, -0.0004] | WITHIN | 0.0047 | -0.0008 | -0.0034 |
| squad | in-domain | 11873 | 0.9092 | 0.9117 | +0.0025 [+0.0005, +0.0045] | WITHIN | 0.9054 | +0.0025 | -0.0013 |
| musique | zero-shot | 2417 | 0.5073 | 0.5053 | -0.0020 [-0.0072, +0.0034] | WITHIN | 0.4734 | -0.0046 | +0.0439 |
| hotpotqa | in-domain | 7405 | 0.9074 | 0.9085 | +0.0011 [-0.0011, +0.0034] | WITHIN | 0.6846 | +0.0026 | +0.0001 |
| 2wiki | in-domain | 12576 | 0.9351 | 0.9339 | -0.0012 [-0.0025, +0.0003] | WITHIN | 0.6079 | -0.0014 | +0.0064 |
| webqsp | zero-shot | 1503 | 0.3204 | 0.3176 | -0.0028 [-0.0087, +0.0031] | WITHIN | 0.0535 | +0.0007 | +0.0060 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8018 | +0.1481 [+0.1424, +0.1533] | GAIN | 0.0047 | +0.1466 | +0.2861 |
| squad | in-domain | 11873 | 0.9100 | 0.9117 | +0.0018 [-0.0006, +0.0042] | WITHIN | 0.9054 | +0.0018 | -0.0031 |
| musique | zero-shot | 2417 | 0.2696 | 0.5053 | +0.2357 [+0.2239, +0.2469] | GAIN | 0.4734 | +0.1229 | +0.3943 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9085 | +0.0070 [+0.0034, +0.0103] | WITHIN | 0.6846 | +0.0099 | +0.0132 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9339 | +0.0607 [+0.0575, +0.0640] | GAIN | 0.6079 | +0.1260 | +0.0713 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3176 | +0.1397 [+0.1169, +0.1644] | GAIN | 0.0535 | +0.1078 | +0.1357 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
