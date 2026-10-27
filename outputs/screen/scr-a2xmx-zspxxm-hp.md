# Screen scr-a2xmx-zspxxm-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g5-zspx-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8039 | 0.8019 | -0.0020 [-0.0031, -0.0011] | WITHIN | 0.0047 | -0.0027 | -0.0072 |
| squad | in-domain | 11873 | 0.9129 | 0.9127 | -0.0003 [-0.0023, +0.0018] | WITHIN | 0.9054 | -0.0003 | -0.0010 |
| musique | in-domain | 2417 | 0.5901 | 0.5863 | -0.0038 [-0.0089, +0.0014] | WITHIN | 0.4734 | -0.0095 | -0.0004 |
| hotpotqa | zero-shot | 7405 | 0.8547 | 0.8515 | -0.0032 [-0.0061, +0.0000] | WITHIN | 0.6846 | -0.0065 | +0.0027 |
| 2wiki | in-domain | 12576 | 0.9312 | 0.9283 | -0.0030 [-0.0048, -0.0013] | WITHIN | 0.6079 | -0.0115 | -0.0009 |
| webqsp | zero-shot | 1503 | 0.2914 | 0.2660 | -0.0254 [-0.0353, -0.0159] | LOSS | 0.0535 | -0.0173 | -0.0027 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8019 | +0.1479 [+0.1424, +0.1532] | GAIN | 0.0047 | +0.1522 | +0.2853 |
| squad | in-domain | 11873 | 0.9116 | 0.9127 | +0.0010 [-0.0013, +0.0033] | WITHIN | 0.9054 | +0.0010 | +0.0008 |
| musique | in-domain | 2417 | 0.5550 | 0.5863 | +0.0313 [+0.0238, +0.0388] | GAIN | 0.4734 | +0.0327 | +0.0546 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8515 | -0.0070 [-0.0117, -0.0023] | WITHIN | 0.6846 | -0.0131 | -0.0211 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9283 | +0.0610 [+0.0577, +0.0645] | GAIN | 0.6079 | +0.1196 | +0.0739 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2660 | +0.1460 [+0.1224, +0.1703] | GAIN | 0.0535 | +0.1098 | +0.0885 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
