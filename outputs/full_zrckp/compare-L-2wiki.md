# Screen L-2wiki (p@swa, s1eval carves)

## Against outputs\full_zrct\fits\L-2wiki (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7793 | 0.8025 | +0.0232 [+0.0203, +0.0260] | GAIN | 0.0047 | +0.0190 | +0.0311 |
| squad | in-domain | 11873 | 0.9121 | 0.9114 | -0.0007 [-0.0025, +0.0013] | WITHIN | 0.9054 | -0.0007 | +0.0040 |
| musique | in-domain | 2417 | 0.5669 | 0.5727 | +0.0058 [-0.0018, +0.0129] | WITHIN | 0.4734 | +0.0000 | +0.0070 |
| hotpotqa | in-domain | 7405 | 0.9001 | 0.9059 | +0.0059 [+0.0027, +0.0090] | WITHIN | 0.6846 | +0.0078 | -0.0015 |
| 2wiki | zero-shot | 12576 | 0.7937 | 0.8541 | +0.0604 [+0.0574, +0.0636] | GAIN | 0.6079 | +0.0761 | +0.0744 |
| webqsp | zero-shot | 1503 | 0.3042 | 0.3686 | +0.0644 [+0.0484, +0.0803] | GAIN | 0.0535 | +0.0585 | +0.0279 |

## Against outputs\full_zrck\fits\L-2wiki (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8008 | 0.8025 | +0.0018 [+0.0008, +0.0027] | WITHIN | 0.0047 | +0.0020 | +0.0046 |
| squad | in-domain | 11873 | 0.9102 | 0.9114 | +0.0012 [-0.0006, +0.0029] | WITHIN | 0.9054 | +0.0012 | +0.0032 |
| musique | in-domain | 2417 | 0.5708 | 0.5727 | +0.0020 [-0.0028, +0.0067] | WITHIN | 0.4734 | +0.0037 | -0.0050 |
| hotpotqa | in-domain | 7405 | 0.9045 | 0.9059 | +0.0014 [-0.0009, +0.0038] | WITHIN | 0.6846 | +0.0036 | -0.0009 |
| 2wiki | zero-shot | 12576 | 0.8576 | 0.8541 | -0.0034 [-0.0056, -0.0014] | WITHIN | 0.6079 | -0.0055 | -0.0047 |
| webqsp | zero-shot | 1503 | 0.3590 | 0.3686 | +0.0096 [+0.0009, +0.0183] | GAIN | 0.0535 | +0.0100 | +0.0106 |

## Against outputs\step1\fits\L-2wiki (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6468 | 0.8025 | +0.1557 [+0.1498, +0.1612] | GAIN | 0.0047 | +0.1570 | +0.2918 |
| squad | in-domain | 11873 | 0.9112 | 0.9114 | +0.0002 [-0.0020, +0.0024] | WITHIN | 0.9054 | +0.0002 | +0.0015 |
| musique | in-domain | 2417 | 0.5601 | 0.5727 | +0.0126 [+0.0048, +0.0203] | GAIN | 0.4734 | +0.0120 | +0.0170 |
| hotpotqa | in-domain | 7405 | 0.8977 | 0.9059 | +0.0082 [+0.0047, +0.0116] | GAIN | 0.6846 | +0.0122 | +0.0090 |
| 2wiki | zero-shot | 12576 | 0.8074 | 0.8541 | +0.0468 [+0.0436, +0.0500] | GAIN | 0.6079 | +0.0508 | +0.0843 |
| webqsp | zero-shot | 1503 | 0.1075 | 0.3686 | +0.2611 [+0.2370, +0.2868] | GAIN | 0.0535 | +0.2149 | +0.0898 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
