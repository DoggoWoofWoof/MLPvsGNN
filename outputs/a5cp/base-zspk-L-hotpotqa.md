# Screen scr-p1gb-zspk-hp (p@swa, s1eval carves)

## Against outputs\step1\fits\L-hotpotqa (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8011 | +0.1471 [+0.1416, +0.1525] | GAIN | 0.0047 | +0.1517 | +0.2815 |
| squad | in-domain | 11873 | 0.9116 | 0.9127 | +0.0010 [-0.0013, +0.0035] | WITHIN | 0.9054 | +0.0010 | +0.0016 |
| musique | in-domain | 2417 | 0.5550 | 0.5662 | +0.0111 [+0.0032, +0.0190] | GAIN | 0.4734 | +0.0149 | +0.0480 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8483 | -0.0102 [-0.0149, -0.0055] | LOSS | 0.6846 | -0.0212 | -0.0385 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9326 | +0.0654 [+0.0621, +0.0689] | GAIN | 0.6079 | +0.1314 | +0.0710 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2808 | +0.1609 [+0.1357, +0.1850] | GAIN | 0.0535 | +0.1184 | +0.0878 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
