# Screen scr-a6ex-zspxl (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g5-zspx (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8030 | 0.8037 | +0.0007 [-0.0003, +0.0018] | WITHIN | 0.0047 | +0.0009 | +0.0012 |
| squad | in-domain | 11873 | 0.9092 | 0.9117 | +0.0025 [+0.0005, +0.0046] | WITHIN | 0.9054 | +0.0025 | +0.0008 |
| musique | zero-shot | 2417 | 0.5073 | 0.4877 | -0.0195 [-0.0262, -0.0131] | LOSS | 0.4734 | -0.0132 | -0.0571 |
| hotpotqa | in-domain | 7405 | 0.9074 | 0.9109 | +0.0035 [+0.0010, +0.0060] | WITHIN | 0.6846 | +0.0065 | +0.0080 |
| 2wiki | in-domain | 12576 | 0.9351 | 0.9296 | -0.0055 [-0.0071, -0.0038] | WITHIN | 0.6079 | -0.0152 | +0.0048 |
| webqsp | zero-shot | 1503 | 0.3204 | 0.3140 | -0.0064 [-0.0159, +0.0024] | WITHIN | 0.0535 | -0.0047 | -0.0073 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8037 | +0.1500 [+0.1444, +0.1554] | GAIN | 0.0047 | +0.1483 | +0.2906 |
| squad | in-domain | 11873 | 0.9100 | 0.9117 | +0.0018 [-0.0007, +0.0042] | WITHIN | 0.9054 | +0.0018 | -0.0011 |
| musique | zero-shot | 2417 | 0.2696 | 0.4877 | +0.2181 [+0.2054, +0.2300] | GAIN | 0.4734 | +0.1142 | +0.2933 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9109 | +0.0094 [+0.0059, +0.0128] | GAIN | 0.6846 | +0.0138 | +0.0211 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9296 | +0.0563 [+0.0531, +0.0597] | GAIN | 0.6079 | +0.1121 | +0.0697 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3140 | +0.1361 [+0.1127, +0.1608] | GAIN | 0.0535 | +0.1025 | +0.1224 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
