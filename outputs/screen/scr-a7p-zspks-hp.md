# Screen scr-a7p-zspks-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1gb-zspk-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8011 | 0.8016 | +0.0005 [-0.0006, +0.0016] | WITHIN | 0.0047 | +0.0000 | +0.0046 |
| squad | in-domain | 11873 | 0.9127 | 0.9142 | +0.0015 [-0.0003, +0.0033] | WITHIN | 0.9054 | +0.0015 | -0.0004 |
| musique | in-domain | 2417 | 0.5662 | 0.5746 | +0.0084 [+0.0031, +0.0135] | GAIN | 0.4734 | +0.0058 | -0.0025 |
| hotpotqa | zero-shot | 7405 | 0.8483 | 0.8533 | +0.0049 [+0.0022, +0.0076] | WITHIN | 0.6846 | +0.0095 | -0.0016 |
| 2wiki | in-domain | 12576 | 0.9326 | 0.9356 | +0.0030 [+0.0015, +0.0046] | WITHIN | 0.6079 | +0.0061 | -0.0077 |
| webqsp | zero-shot | 1503 | 0.2808 | 0.2851 | +0.0042 [-0.0056, +0.0141] | WITHIN | 0.0535 | +0.0053 | +0.0106 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8016 | +0.1476 [+0.1422, +0.1531] | GAIN | 0.0047 | +0.1517 | +0.2861 |
| squad | in-domain | 11873 | 0.9116 | 0.9142 | +0.0025 [+0.0003, +0.0048] | WITHIN | 0.9054 | +0.0025 | +0.0012 |
| musique | in-domain | 2417 | 0.5550 | 0.5746 | +0.0195 [+0.0119, +0.0272] | GAIN | 0.4734 | +0.0207 | +0.0455 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8533 | -0.0053 [-0.0097, -0.0008] | WITHIN | 0.6846 | -0.0117 | -0.0401 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9356 | +0.0684 [+0.0652, +0.0719] | GAIN | 0.6079 | +0.1375 | +0.0633 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2851 | +0.1651 [+0.1413, +0.1888] | GAIN | 0.0535 | +0.1238 | +0.0985 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
