# Screen scr-zgf-hp-off (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrm-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7801 | 0.7649 | -0.0152 [-0.0170, -0.0135] | LOSS | 0.0047 | -0.0158 | -0.0473 |
| squad | in-domain | 11873 | 0.9116 | 0.9116 | +0.0000 [-0.0020, +0.0020] | WITHIN | 0.9054 | +0.0000 | -0.0021 |
| musique | in-domain | 2417 | 0.5600 | 0.5640 | +0.0041 [-0.0020, +0.0104] | WITHIN | 0.4734 | -0.0004 | -0.0079 |
| hotpotqa | zero-shot | 7405 | 0.8457 | 0.8408 | -0.0049 [-0.0084, -0.0014] | WITHIN | 0.6846 | -0.0122 | +0.0014 |
| 2wiki | in-domain | 12576 | 0.8681 | 0.8588 | -0.0093 [-0.0115, -0.0070] | LOSS | 0.6079 | -0.0246 | +0.0118 |
| webqsp | zero-shot | 1503 | 0.2416 | 0.2339 | -0.0077 [-0.0174, +0.0019] | WITHIN | 0.0535 | -0.0067 | -0.0153 |

## Against outputs\screen\fits\scr-zgf-hp (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7815 | 0.7649 | -0.0167 [-0.0187, -0.0147] | LOSS | 0.0047 | -0.0144 | -0.0554 |
| squad | in-domain | 11873 | 0.8972 | 0.9116 | +0.0143 [+0.0110, +0.0175] | GAIN | 0.9054 | +0.0143 | +0.0264 |
| musique | in-domain | 2417 | 0.5001 | 0.5640 | +0.0640 [+0.0558, +0.0723] | GAIN | 0.4734 | +0.0658 | +0.1283 |
| hotpotqa | zero-shot | 7405 | 0.8020 | 0.8408 | +0.0388 [+0.0343, +0.0430] | GAIN | 0.6846 | +0.0689 | +0.0583 |
| 2wiki | in-domain | 12576 | 0.8626 | 0.8588 | -0.0038 [-0.0063, -0.0013] | WITHIN | 0.6079 | -0.0151 | +0.0476 |
| webqsp | zero-shot | 1503 | 0.2407 | 0.2339 | -0.0069 [-0.0167, +0.0027] | WITHIN | 0.0535 | -0.0033 | -0.0193 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
