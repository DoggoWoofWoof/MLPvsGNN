# Screen scr-ztop50 (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.6361 | -0.0176 [-0.0212, -0.0140] | LOSS | 0.0047 | -0.0182 | -0.0145 |
| squad | in-domain | 11873 | 0.9100 | 0.9112 | +0.0013 [-0.0009, +0.0035] | WITHIN | 0.9054 | +0.0013 | -0.0019 |
| musique | zero-shot | 2417 | 0.2696 | 0.4083 | +0.1387 [+0.1279, +0.1492] | GAIN | 0.4734 | +0.0736 | +0.1758 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9019 | +0.0004 [-0.0024, +0.0030] | WITHIN | 0.6846 | +0.0004 | +0.0076 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8691 | -0.0041 [-0.0061, -0.0021] | WITHIN | 0.6079 | -0.0083 | -0.0027 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.1954 | +0.0175 [+0.0035, +0.0321] | GAIN | 0.0535 | +0.0186 | +0.0106 |

## Against outputs\screen\fits\scr-zret (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6496 | 0.6361 | -0.0135 [-0.0170, -0.0100] | LOSS | 0.0047 | -0.0127 | -0.0134 |
| squad | in-domain | 11873 | 0.9106 | 0.9112 | +0.0006 [-0.0014, +0.0027] | WITHIN | 0.9054 | +0.0006 | +0.0020 |
| musique | zero-shot | 2417 | 0.3897 | 0.4083 | +0.0185 [+0.0101, +0.0265] | GAIN | 0.4734 | +0.0054 | +0.0219 |
| hotpotqa | in-domain | 7405 | 0.9061 | 0.9019 | -0.0042 [-0.0068, -0.0016] | WITHIN | 0.6846 | -0.0072 | +0.0042 |
| 2wiki | in-domain | 12576 | 0.8714 | 0.8691 | -0.0023 [-0.0044, -0.0003] | WITHIN | 0.6079 | -0.0028 | -0.0068 |
| webqsp | zero-shot | 1503 | 0.1766 | 0.1954 | +0.0188 [+0.0078, +0.0296] | GAIN | 0.0535 | +0.0180 | +0.0160 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
