# Screen scr-a4fx-zspxf-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g5-zspx-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8039 | 0.8014 | -0.0025 [-0.0039, -0.0012] | WITHIN | 0.0047 | -0.0044 | -0.0160 |
| squad | in-domain | 11873 | 0.9129 | 0.9119 | -0.0010 [-0.0029, +0.0009] | WITHIN | 0.9054 | -0.0010 | -0.0040 |
| musique | in-domain | 2417 | 0.5901 | 0.5852 | -0.0049 [-0.0100, +0.0003] | WITHIN | 0.4734 | -0.0108 | -0.0066 |
| hotpotqa | zero-shot | 7405 | 0.8547 | 0.8575 | +0.0028 [-0.0005, +0.0063] | WITHIN | 0.6846 | +0.0069 | -0.0061 |
| 2wiki | in-domain | 12576 | 0.9312 | 0.9269 | -0.0043 [-0.0060, -0.0027] | WITHIN | 0.6079 | -0.0142 | -0.0052 |
| webqsp | zero-shot | 1503 | 0.2914 | 0.3001 | +0.0087 [-0.0009, +0.0187] | WITHIN | 0.0535 | +0.0060 | +0.0133 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8014 | +0.1474 [+0.1419, +0.1528] | GAIN | 0.0047 | +0.1504 | +0.2764 |
| squad | in-domain | 11873 | 0.9116 | 0.9119 | +0.0003 [-0.0020, +0.0026] | WITHIN | 0.9054 | +0.0003 | -0.0021 |
| musique | in-domain | 2417 | 0.5550 | 0.5852 | +0.0302 [+0.0224, +0.0380] | GAIN | 0.4734 | +0.0314 | +0.0484 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8575 | -0.0011 [-0.0060, +0.0038] | WITHIN | 0.6846 | +0.0003 | -0.0298 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9269 | +0.0597 [+0.0563, +0.0633] | GAIN | 0.6079 | +0.1169 | +0.0696 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3001 | +0.1801 [+0.1556, +0.2038] | GAIN | 0.0535 | +0.1331 | +0.1045 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
