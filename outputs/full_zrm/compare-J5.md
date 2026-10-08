# Screen J5 (p@swa, s1eval carves)

## Against outputs\full_zret\fits\J5 (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6431 | 0.7788 | +0.1357 [+0.1301, +0.1412] | GAIN | 0.0047 | +0.1401 | +0.2694 |
| squad | in-domain | 11873 | 0.9109 | 0.9108 | -0.0001 [-0.0022, +0.0021] | WITHIN | 0.9054 | -0.0001 | +0.0007 |
| musique | in-domain | 2417 | 0.5640 | 0.5665 | +0.0026 [-0.0031, +0.0080] | WITHIN | 0.4734 | -0.0008 | +0.0029 |
| hotpotqa | in-domain | 7405 | 0.9008 | 0.8997 | -0.0011 [-0.0040, +0.0015] | WITHIN | 0.6846 | -0.0005 | +0.0030 |
| 2wiki | in-domain | 12576 | 0.8659 | 0.8705 | +0.0046 [+0.0026, +0.0066] | WITHIN | 0.6079 | +0.0072 | -0.0021 |
| webqsp | zero-shot | 1503 | 0.1417 | 0.2558 | +0.1140 [+0.0927, +0.1366] | GAIN | 0.0535 | +0.0852 | +0.0699 |

## Against outputs\full_rmatch\fits\J5 (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7788 | 0.7788 | +0.0001 [-0.0011, +0.0012] | WITHIN | 0.0047 | +0.0000 | +0.0016 |
| squad | in-domain | 11873 | 0.9075 | 0.9108 | +0.0033 [+0.0012, +0.0056] | WITHIN | 0.9054 | +0.0033 | +0.0018 |
| musique | in-domain | 2417 | 0.5685 | 0.5665 | -0.0020 [-0.0077, +0.0034] | WITHIN | 0.4734 | -0.0058 | +0.0004 |
| hotpotqa | in-domain | 7405 | 0.9026 | 0.8997 | -0.0030 [-0.0056, -0.0003] | WITHIN | 0.6846 | -0.0049 | +0.0035 |
| 2wiki | in-domain | 12576 | 0.8730 | 0.8705 | -0.0025 [-0.0045, -0.0005] | WITHIN | 0.6079 | -0.0053 | +0.0004 |
| webqsp | zero-shot | 1503 | 0.2795 | 0.2558 | -0.0237 [-0.0355, -0.0116] | LOSS | 0.0535 | -0.0193 | -0.0166 |

## Against outputs\step1\fits\J5 (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6443 | 0.7788 | +0.1346 [+0.1292, +0.1401] | GAIN | 0.0047 | +0.1409 | +0.2556 |
| squad | in-domain | 11873 | 0.9095 | 0.9108 | +0.0013 [-0.0009, +0.0035] | WITHIN | 0.9054 | +0.0013 | -0.0013 |
| musique | in-domain | 2417 | 0.5609 | 0.5665 | +0.0057 [-0.0004, +0.0116] | WITHIN | 0.4734 | +0.0103 | +0.0066 |
| hotpotqa | in-domain | 7405 | 0.9013 | 0.8997 | -0.0016 [-0.0047, +0.0014] | WITHIN | 0.6846 | -0.0019 | +0.0078 |
| 2wiki | in-domain | 12576 | 0.8694 | 0.8705 | +0.0011 [-0.0010, +0.0031] | WITHIN | 0.6079 | -0.0001 | -0.0013 |
| webqsp | zero-shot | 1503 | 0.1167 | 0.2558 | +0.1390 [+0.1185, +0.1615] | GAIN | 0.0535 | +0.1084 | +0.0725 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
