# Screen scr-r2t-zg6m-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zg6-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8115 | 0.8113 | -0.0002 [-0.0013, +0.0009] | WITHIN | 0.0047 | -0.0008 | -0.0026 |
| squad | in-domain | 11873 | 0.9122 | 0.9079 | -0.0043 [-0.0066, -0.0020] | WITHIN | 0.9054 | -0.0043 | -0.0067 |
| musique | in-domain | 2417 | 0.5953 | 0.5912 | -0.0041 [-0.0100, +0.0024] | WITHIN | 0.4734 | -0.0062 | -0.0236 |
| hotpotqa | zero-shot | 7405 | 0.8391 | 0.8380 | -0.0011 [-0.0043, +0.0022] | WITHIN | 0.6846 | +0.0035 | +0.0030 |
| 2wiki | in-domain | 12576 | 0.9463 | 0.9434 | -0.0030 [-0.0045, -0.0014] | WITHIN | 0.6079 | -0.0103 | +0.0043 |
| webqsp | zero-shot | 1503 | 0.2430 | 0.1825 | -0.0605 [-0.0778, -0.0424] | LOSS | 0.0535 | -0.0479 | -0.0459 |

## Against outputs\screen\fits\scr-kg-base-zspk-hp (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8037 | 0.8113 | +0.0076 [+0.0059, +0.0093] | GAIN | 0.0047 | +0.0084 | +0.0249 |
| squad | in-domain | 11873 | 0.9139 | 0.9079 | -0.0060 [-0.0084, -0.0036] | WITHIN | 0.9054 | -0.0060 | -0.0013 |
| musique | in-domain | 2417 | 0.5754 | 0.5912 | +0.0158 [+0.0089, +0.0224] | GAIN | 0.4734 | +0.0265 | -0.0079 |
| hotpotqa | zero-shot | 7405 | 0.8550 | 0.8380 | -0.0169 [-0.0209, -0.0128] | LOSS | 0.6846 | -0.0146 | -0.0300 |
| 2wiki | in-domain | 12576 | 0.9265 | 0.9434 | +0.0168 [+0.0147, +0.0189] | GAIN | 0.6079 | +0.0513 | -0.0067 |
| webqsp | zero-shot | 1503 | 0.2640 | 0.1825 | -0.0815 [-0.1026, -0.0611] | LOSS | 0.0535 | -0.0625 | -0.0373 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
