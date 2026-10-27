# Screen scr-a4fx-zspxf (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g5-zspx (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8030 | 0.7996 | -0.0034 [-0.0048, -0.0021] | WITHIN | 0.0047 | -0.0040 | -0.0164 |
| squad | in-domain | 11873 | 0.9092 | 0.9081 | -0.0011 [-0.0033, +0.0011] | WITHIN | 0.9054 | -0.0011 | -0.0045 |
| musique | zero-shot | 2417 | 0.5073 | 0.3134 | -0.1938 [-0.2051, -0.1830] | LOSS | 0.4734 | -0.1134 | -0.3471 |
| hotpotqa | in-domain | 7405 | 0.9074 | 0.9068 | -0.0006 [-0.0028, +0.0018] | WITHIN | 0.6846 | +0.0001 | -0.0059 |
| 2wiki | in-domain | 12576 | 0.9351 | 0.9310 | -0.0041 [-0.0056, -0.0025] | WITHIN | 0.6079 | -0.0136 | +0.0069 |
| webqsp | zero-shot | 1503 | 0.3204 | 0.3147 | -0.0057 [-0.0176, +0.0058] | WITHIN | 0.0535 | -0.0067 | -0.0200 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7996 | +0.1458 [+0.1403, +0.1510] | GAIN | 0.0047 | +0.1434 | +0.2731 |
| squad | in-domain | 11873 | 0.9100 | 0.9081 | -0.0019 [-0.0044, +0.0007] | WITHIN | 0.9054 | -0.0019 | -0.0063 |
| musique | zero-shot | 2417 | 0.2696 | 0.3134 | +0.0439 [+0.0328, +0.0543] | GAIN | 0.4734 | +0.0141 | +0.0033 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9068 | +0.0053 [+0.0018, +0.0086] | WITHIN | 0.6846 | +0.0074 | +0.0072 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9310 | +0.0577 [+0.0547, +0.0609] | GAIN | 0.6079 | +0.1137 | +0.0719 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3147 | +0.1368 [+0.1137, +0.1605] | GAIN | 0.0535 | +0.1005 | +0.1098 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
