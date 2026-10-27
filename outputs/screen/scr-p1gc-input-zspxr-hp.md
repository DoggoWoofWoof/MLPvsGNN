# Screen scr-p1gc-zspxr-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1gc-zspx-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7996 | 0.8006 | +0.0010 [+0.0000, +0.0020] | WITHIN | 0.0047 | +0.0021 | +0.0059 |
| squad | in-domain | 11873 | 0.9120 | 0.9137 | +0.0017 [+0.0000, +0.0033] | WITHIN | 0.9054 | +0.0017 | +0.0024 |
| musique | in-domain | 2417 | 0.5814 | 0.5839 | +0.0025 [-0.0026, +0.0076] | WITHIN | 0.4734 | +0.0079 | +0.0021 |
| hotpotqa | zero-shot | 7405 | 0.8575 | 0.8519 | -0.0055 [-0.0084, -0.0028] | WITHIN | 0.6846 | -0.0100 | +0.0042 |
| 2wiki | in-domain | 12576 | 0.9338 | 0.9346 | +0.0007 [-0.0009, +0.0023] | WITHIN | 0.6079 | -0.0010 | +0.0046 |
| webqsp | zero-shot | 1503 | 0.3072 | 0.2715 | -0.0357 [-0.0453, -0.0259] | LOSS | 0.0535 | -0.0279 | -0.0093 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8006 | +0.1466 [+0.1410, +0.1521] | GAIN | 0.0047 | +0.1519 | +0.2873 |
| squad | in-domain | 11873 | 0.9116 | 0.9137 | +0.0020 [-0.0002, +0.0045] | WITHIN | 0.9054 | +0.0020 | +0.0004 |
| musique | in-domain | 2417 | 0.5550 | 0.5839 | +0.0289 [+0.0214, +0.0365] | GAIN | 0.4734 | +0.0331 | +0.0439 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8519 | -0.0066 [-0.0112, -0.0022] | WITHIN | 0.6846 | -0.0146 | -0.0093 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9346 | +0.0674 [+0.0641, +0.0708] | GAIN | 0.6079 | +0.1377 | +0.0674 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2715 | +0.1515 [+0.1270, +0.1751] | GAIN | 0.0535 | +0.1111 | +0.0912 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
