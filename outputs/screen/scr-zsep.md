# Screen scr-zsep (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zret (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6496 | 0.6540 | +0.0043 [+0.0016, +0.0073] | WITHIN | 0.0047 | +0.0025 | +0.0169 |
| squad | in-domain | 11873 | 0.9106 | 0.9095 | -0.0012 [-0.0031, +0.0008] | WITHIN | 0.9054 | -0.0012 | +0.0016 |
| musique | zero-shot | 2417 | 0.3897 | 0.3210 | -0.0687 [-0.0765, -0.0610] | LOSS | 0.4734 | -0.0401 | -0.0530 |
| hotpotqa | in-domain | 7405 | 0.9061 | 0.9031 | -0.0030 [-0.0054, -0.0004] | WITHIN | 0.6846 | -0.0068 | +0.0205 |
| 2wiki | in-domain | 12576 | 0.8714 | 0.8708 | -0.0007 [-0.0025, +0.0013] | WITHIN | 0.6079 | -0.0035 | +0.0151 |
| webqsp | zero-shot | 1503 | 0.1766 | 0.1770 | +0.0003 [-0.0095, +0.0096] | WITHIN | 0.0535 | +0.0020 | -0.0013 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.6540 | +0.0002 [-0.0029, +0.0033] | WITHIN | 0.0047 | -0.0031 | +0.0157 |
| squad | in-domain | 11873 | 0.9100 | 0.9095 | -0.0005 [-0.0029, +0.0018] | WITHIN | 0.9054 | -0.0005 | -0.0024 |
| musique | zero-shot | 2417 | 0.2696 | 0.3210 | +0.0514 [+0.0424, +0.0603] | GAIN | 0.4734 | +0.0281 | +0.1010 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9031 | +0.0016 [-0.0011, +0.0043] | WITHIN | 0.6846 | +0.0008 | +0.0239 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8708 | -0.0024 [-0.0046, -0.0004] | WITHIN | 0.6079 | -0.0090 | +0.0192 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.1770 | -0.0009 [-0.0125, +0.0120] | WITHIN | 0.0535 | +0.0027 | -0.0067 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
