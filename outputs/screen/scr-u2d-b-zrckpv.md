# Screen scr-u2d-new-zrckpv (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-new-zrckp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7999 | 0.7986 | -0.0013 [-0.0022, -0.0004] | WITHIN | 0.0047 | -0.0013 | -0.0058 |
| squad | in-domain | 11873 | 0.9129 | 0.9114 | -0.0015 [-0.0034, +0.0003] | WITHIN | 0.9054 | -0.0015 | +0.0009 |
| musique | zero-shot | 2417 | 0.5189 | 0.5195 | +0.0007 [-0.0040, +0.0056] | WITHIN | 0.4734 | +0.0050 | +0.0000 |
| hotpotqa | in-domain | 7405 | 0.8484 | 0.8450 | -0.0034 [-0.0059, -0.0010] | WITHIN | 0.6846 | -0.0081 | -0.0001 |
| 2wiki | in-domain | 12576 | 0.9204 | 0.9147 | -0.0057 [-0.0075, -0.0042] | WITHIN | 0.6079 | -0.0146 | -0.0002 |
| webqsp | zero-shot | 1503 | 0.3951 | 0.3729 | -0.0222 [-0.0327, -0.0117] | LOSS | 0.0535 | -0.0160 | -0.0180 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
