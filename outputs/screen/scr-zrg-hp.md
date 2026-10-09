# Screen scr-zrg-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrm-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7801 | 0.7803 | +0.0003 [-0.0008, +0.0013] | WITHIN | 0.0047 | -0.0001 | +0.0048 |
| squad | in-domain | 11873 | 0.9116 | 0.9123 | +0.0008 [-0.0013, +0.0028] | WITHIN | 0.9054 | +0.0008 | -0.0008 |
| musique | in-domain | 2417 | 0.5600 | 0.5687 | +0.0087 [+0.0032, +0.0142] | GAIN | 0.4734 | +0.0137 | -0.0141 |
| hotpotqa | zero-shot | 7405 | 0.8457 | 0.8363 | -0.0094 [-0.0129, -0.0059] | LOSS | 0.6846 | -0.0186 | -0.0038 |
| 2wiki | in-domain | 12576 | 0.8681 | 0.8734 | +0.0053 [+0.0032, +0.0075] | WITHIN | 0.6079 | +0.0169 | +0.0046 |
| webqsp | zero-shot | 1503 | 0.2416 | 0.2497 | +0.0081 [-0.0018, +0.0179] | WITHIN | 0.0535 | +0.0086 | +0.0013 |

## Against outputs\screen\fits\scr-zgs-hp (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6497 | 0.7803 | +0.1306 [+0.1253, +0.1361] | GAIN | 0.0047 | +0.1354 | +0.2633 |
| squad | in-domain | 11873 | 0.9117 | 0.9123 | +0.0006 [-0.0017, +0.0027] | WITHIN | 0.9054 | +0.0006 | +0.0025 |
| musique | in-domain | 2417 | 0.5558 | 0.5687 | +0.0129 [+0.0072, +0.0186] | GAIN | 0.4734 | +0.0232 | +0.0037 |
| hotpotqa | zero-shot | 7405 | 0.8392 | 0.8363 | -0.0029 [-0.0065, +0.0007] | WITHIN | 0.6846 | -0.0057 | +0.0031 |
| 2wiki | in-domain | 12576 | 0.8736 | 0.8734 | -0.0002 [-0.0023, +0.0018] | WITHIN | 0.6079 | -0.0011 | +0.0171 |
| webqsp | zero-shot | 1503 | 0.1227 | 0.2497 | +0.1270 [+0.1062, +0.1484] | GAIN | 0.0535 | +0.1005 | +0.0612 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7803 | +0.1264 [+0.1211, +0.1315] | GAIN | 0.0047 | +0.1338 | +0.2604 |
| squad | in-domain | 11873 | 0.9116 | 0.9123 | +0.0007 [-0.0016, +0.0029] | WITHIN | 0.9054 | +0.0007 | +0.0012 |
| musique | in-domain | 2417 | 0.5550 | 0.5687 | +0.0137 [+0.0076, +0.0203] | GAIN | 0.4734 | +0.0314 | -0.0017 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8363 | -0.0222 [-0.0263, -0.0182] | LOSS | 0.6846 | -0.0421 | +0.0005 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8734 | +0.0062 [+0.0038, +0.0084] | WITHIN | 0.6079 | +0.0197 | -0.0018 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2497 | +0.1297 [+0.1083, +0.1512] | GAIN | 0.0535 | +0.1018 | +0.0645 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
