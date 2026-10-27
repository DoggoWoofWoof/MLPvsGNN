# Screen scr-p1gc-zspx-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1gb-zspk-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8011 | 0.7996 | -0.0015 [-0.0025, -0.0005] | WITHIN | 0.0047 | -0.0019 | -0.0001 |
| squad | in-domain | 11873 | 0.9127 | 0.9120 | -0.0007 [-0.0025, +0.0013] | WITHIN | 0.9054 | -0.0007 | -0.0036 |
| musique | in-domain | 2417 | 0.5662 | 0.5814 | +0.0152 [+0.0095, +0.0212] | GAIN | 0.4734 | +0.0103 | -0.0062 |
| hotpotqa | zero-shot | 7405 | 0.8483 | 0.8575 | +0.0091 [+0.0063, +0.0122] | GAIN | 0.6846 | +0.0166 | +0.0250 |
| 2wiki | in-domain | 12576 | 0.9326 | 0.9338 | +0.0013 [-0.0003, +0.0028] | WITHIN | 0.6079 | +0.0074 | -0.0083 |
| webqsp | zero-shot | 1503 | 0.2808 | 0.3072 | +0.0264 [+0.0161, +0.0366] | GAIN | 0.0535 | +0.0206 | +0.0126 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7996 | +0.1456 [+0.1401, +0.1510] | GAIN | 0.0047 | +0.1497 | +0.2813 |
| squad | in-domain | 11873 | 0.9116 | 0.9120 | +0.0003 [-0.0019, +0.0026] | WITHIN | 0.9054 | +0.0003 | -0.0020 |
| musique | in-domain | 2417 | 0.5550 | 0.5814 | +0.0264 [+0.0189, +0.0341] | GAIN | 0.4734 | +0.0252 | +0.0418 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8575 | -0.0011 [-0.0058, +0.0036] | WITHIN | 0.6846 | -0.0046 | -0.0135 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9338 | +0.0666 [+0.0633, +0.0701] | GAIN | 0.6079 | +0.1388 | +0.0627 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3072 | +0.1872 [+0.1626, +0.2113] | GAIN | 0.0535 | +0.1391 | +0.1005 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
