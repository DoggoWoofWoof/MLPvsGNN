# Screen scr-zgs (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zret (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6496 | 0.6527 | +0.0030 [+0.0001, +0.0059] | WITHIN | 0.0047 | +0.0015 | +0.0024 |
| squad | in-domain | 11873 | 0.9106 | 0.9116 | +0.0009 [-0.0011, +0.0029] | WITHIN | 0.9054 | +0.0009 | +0.0029 |
| musique | zero-shot | 2417 | 0.3897 | 0.3798 | -0.0100 [-0.0165, -0.0032] | LOSS | 0.4734 | -0.0112 | -0.0165 |
| hotpotqa | in-domain | 7405 | 0.9061 | 0.9047 | -0.0014 [-0.0038, +0.0012] | WITHIN | 0.6846 | -0.0038 | +0.0084 |
| 2wiki | in-domain | 12576 | 0.8714 | 0.8719 | +0.0005 [-0.0014, +0.0024] | WITHIN | 0.6079 | +0.0048 | -0.0019 |
| webqsp | zero-shot | 1503 | 0.1766 | 0.1875 | +0.0109 [+0.0024, +0.0191] | GAIN | 0.0535 | +0.0093 | +0.0007 |

## Against outputs\screen\fits\scr-gsurg (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6549 | 0.6527 | -0.0023 [-0.0054, +0.0009] | WITHIN | 0.0047 | -0.0036 | -0.0119 |
| squad | in-domain | 11873 | 0.9089 | 0.9116 | +0.0027 [+0.0005, +0.0048] | WITHIN | 0.9054 | +0.0027 | +0.0021 |
| musique | zero-shot | 2417 | 0.3537 | 0.3798 | +0.0260 [+0.0168, +0.0346] | GAIN | 0.4734 | +0.0190 | +0.0699 |
| hotpotqa | in-domain | 7405 | 0.9018 | 0.9047 | +0.0029 [+0.0003, +0.0055] | WITHIN | 0.6846 | +0.0041 | +0.0070 |
| 2wiki | in-domain | 12576 | 0.8746 | 0.8719 | -0.0027 [-0.0047, -0.0006] | WITHIN | 0.6079 | +0.0009 | +0.0010 |
| webqsp | zero-shot | 1503 | 0.1771 | 0.1875 | +0.0104 [-0.0006, +0.0216] | WITHIN | 0.0535 | +0.0106 | +0.0040 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.6527 | -0.0011 [-0.0042, +0.0019] | WITHIN | 0.0047 | -0.0040 | +0.0012 |
| squad | in-domain | 11873 | 0.9100 | 0.9116 | +0.0016 [-0.0007, +0.0040] | WITHIN | 0.9054 | +0.0016 | -0.0010 |
| musique | zero-shot | 2417 | 0.2696 | 0.3798 | +0.1102 [+0.1003, +0.1198] | GAIN | 0.4734 | +0.0571 | +0.1374 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9047 | +0.0032 [+0.0006, +0.0058] | WITHIN | 0.6846 | +0.0038 | +0.0117 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8719 | -0.0013 [-0.0033, +0.0008] | WITHIN | 0.6079 | -0.0007 | +0.0022 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.1875 | +0.0096 [-0.0024, +0.0219] | WITHIN | 0.0535 | +0.0100 | -0.0047 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
