# Screen scr-zgs-hp (p@swa, s1eval carves)

## Against outputs\full_zret\fits\L-hotpotqa (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6419 | 0.6497 | +0.0078 [+0.0047, +0.0109] | GAIN | 0.0047 | +0.0084 | +0.0066 |
| squad | in-domain | 11873 | 0.9127 | 0.9117 | -0.0010 [-0.0032, +0.0011] | WITHIN | 0.9054 | -0.0010 | -0.0027 |
| musique | in-domain | 2417 | 0.5623 | 0.5558 | -0.0065 [-0.0127, -0.0008] | WITHIN | 0.4734 | -0.0091 | -0.0182 |
| hotpotqa | zero-shot | 7405 | 0.8445 | 0.8392 | -0.0053 [-0.0087, -0.0016] | WITHIN | 0.6846 | -0.0105 | -0.0031 |
| 2wiki | in-domain | 12576 | 0.8728 | 0.8736 | +0.0009 [-0.0012, +0.0030] | WITHIN | 0.6079 | +0.0050 | -0.0154 |
| webqsp | zero-shot | 1503 | 0.1195 | 0.1227 | +0.0032 [-0.0046, +0.0117] | WITHIN | 0.0535 | +0.0033 | +0.0080 |

## Against outputs\screen\fits\scr-gsurg-hp (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6556 | 0.6497 | -0.0059 [-0.0092, -0.0028] | WITHIN | 0.0047 | -0.0051 | -0.0117 |
| squad | in-domain | 11873 | 0.9097 | 0.9117 | +0.0020 [-0.0004, +0.0044] | WITHIN | 0.9054 | +0.0020 | +0.0020 |
| musique | in-domain | 2417 | 0.5599 | 0.5558 | -0.0041 [-0.0109, +0.0023] | WITHIN | 0.4734 | -0.0091 | -0.0012 |
| hotpotqa | zero-shot | 7405 | 0.8438 | 0.8392 | -0.0046 [-0.0085, -0.0005] | WITHIN | 0.6846 | -0.0085 | +0.0047 |
| 2wiki | in-domain | 12576 | 0.8763 | 0.8736 | -0.0027 [-0.0048, -0.0005] | WITHIN | 0.6079 | -0.0049 | -0.0083 |
| webqsp | zero-shot | 1503 | 0.1324 | 0.1227 | -0.0096 [-0.0198, +0.0003] | WITHIN | 0.0535 | -0.0080 | +0.0073 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.6497 | -0.0043 [-0.0077, -0.0009] | WITHIN | 0.0047 | -0.0016 | -0.0029 |
| squad | in-domain | 11873 | 0.9116 | 0.9117 | +0.0001 [-0.0020, +0.0023] | WITHIN | 0.9054 | +0.0001 | -0.0013 |
| musique | in-domain | 2417 | 0.5550 | 0.5558 | +0.0008 [-0.0053, +0.0073] | WITHIN | 0.4734 | +0.0083 | -0.0054 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8392 | -0.0193 [-0.0234, -0.0151] | LOSS | 0.6846 | -0.0365 | -0.0026 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8736 | +0.0064 [+0.0040, +0.0088] | WITHIN | 0.6079 | +0.0208 | -0.0189 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.1227 | +0.0028 [-0.0064, +0.0119] | WITHIN | 0.0535 | +0.0013 | +0.0033 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
