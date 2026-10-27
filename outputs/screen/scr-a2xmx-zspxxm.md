# Screen scr-a2xmx-zspxxm (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g5-zspx (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8030 | 0.8004 | -0.0026 [-0.0036, -0.0016] | WITHIN | 0.0047 | -0.0016 | -0.0073 |
| squad | in-domain | 11873 | 0.9092 | 0.9116 | +0.0024 [+0.0003, +0.0045] | WITHIN | 0.9054 | +0.0024 | -0.0024 |
| musique | zero-shot | 2417 | 0.5073 | 0.4949 | -0.0124 [-0.0199, -0.0054] | LOSS | 0.4734 | -0.0091 | +0.0211 |
| hotpotqa | in-domain | 7405 | 0.9074 | 0.9080 | +0.0006 [-0.0017, +0.0029] | WITHIN | 0.6846 | +0.0018 | -0.0045 |
| 2wiki | in-domain | 12576 | 0.9351 | 0.9343 | -0.0008 [-0.0022, +0.0007] | WITHIN | 0.6079 | -0.0005 | +0.0058 |
| webqsp | zero-shot | 1503 | 0.3204 | 0.3291 | +0.0088 [-0.0006, +0.0183] | WITHIN | 0.0535 | +0.0053 | +0.0086 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8004 | +0.1467 [+0.1410, +0.1520] | GAIN | 0.0047 | +0.1457 | +0.2822 |
| squad | in-domain | 11873 | 0.9100 | 0.9116 | +0.0017 [-0.0007, +0.0039] | WITHIN | 0.9054 | +0.0017 | -0.0043 |
| musique | zero-shot | 2417 | 0.2696 | 0.4949 | +0.2253 [+0.2130, +0.2369] | GAIN | 0.4734 | +0.1183 | +0.3715 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9080 | +0.0065 [+0.0031, +0.0098] | WITHIN | 0.6846 | +0.0090 | +0.0086 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9343 | +0.0610 [+0.0579, +0.0642] | GAIN | 0.6079 | +0.1268 | +0.0708 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3291 | +0.1513 [+0.1278, +0.1760] | GAIN | 0.0535 | +0.1124 | +0.1384 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
