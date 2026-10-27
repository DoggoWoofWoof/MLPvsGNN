# Screen scr-a3vp-zrckpy (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1gb-zrckp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7993 | 0.8004 | +0.0012 [+0.0001, +0.0022] | WITHIN | 0.0047 | +0.0015 | +0.0119 |
| squad | in-domain | 11873 | 0.9127 | 0.9116 | -0.0010 [-0.0028, +0.0008] | WITHIN | 0.9054 | -0.0010 | -0.0008 |
| musique | zero-shot | 2417 | 0.5013 | 0.5067 | +0.0054 [+0.0010, +0.0098] | WITHIN | 0.4734 | +0.0054 | +0.0062 |
| hotpotqa | in-domain | 7405 | 0.9035 | 0.9042 | +0.0007 [-0.0016, +0.0030] | WITHIN | 0.6846 | -0.0007 | +0.0065 |
| 2wiki | in-domain | 12576 | 0.9377 | 0.9308 | -0.0069 [-0.0084, -0.0054] | WITHIN | 0.6079 | -0.0192 | +0.0051 |
| webqsp | zero-shot | 1503 | 0.3768 | 0.3655 | -0.0113 [-0.0206, -0.0014] | LOSS | 0.0535 | -0.0173 | -0.0140 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8004 | +0.1467 [+0.1413, +0.1519] | GAIN | 0.0047 | +0.1462 | +0.2802 |
| squad | in-domain | 11873 | 0.9100 | 0.9116 | +0.0017 [-0.0007, +0.0040] | WITHIN | 0.9054 | +0.0017 | -0.0030 |
| musique | zero-shot | 2417 | 0.2696 | 0.5067 | +0.2371 [+0.2254, +0.2492] | GAIN | 0.4734 | +0.1216 | +0.4261 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9042 | +0.0027 [-0.0007, +0.0062] | WITHIN | 0.6846 | +0.0007 | -0.0184 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9308 | +0.0576 [+0.0545, +0.0608] | GAIN | 0.6079 | +0.1155 | +0.0761 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3655 | +0.1876 [+0.1655, +0.2107] | GAIN | 0.0535 | +0.1537 | +0.0805 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
