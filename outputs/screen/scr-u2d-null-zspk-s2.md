# Screen scr-u2d-base-zspk-s2 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zspk (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8021 | 0.8039 | +0.0018 [+0.0007, +0.0029] | WITHIN | 0.0047 | +0.0027 | +0.0028 |
| squad | in-domain | 11873 | 0.9118 | 0.9110 | -0.0008 [-0.0028, +0.0011] | WITHIN | 0.9054 | -0.0008 | +0.0050 |
| musique | zero-shot | 2417 | 0.5143 | 0.5124 | -0.0019 [-0.0070, +0.0029] | WITHIN | 0.4734 | -0.0054 | -0.0186 |
| hotpotqa | in-domain | 7405 | 0.9088 | 0.9085 | -0.0003 [-0.0024, +0.0018] | WITHIN | 0.6846 | +0.0007 | +0.0053 |
| 2wiki | in-domain | 12576 | 0.9346 | 0.9358 | +0.0012 [-0.0003, +0.0027] | WITHIN | 0.6079 | +0.0055 | -0.0041 |
| webqsp | zero-shot | 1503 | 0.3162 | 0.3141 | -0.0022 [-0.0111, +0.0065] | WITHIN | 0.0535 | -0.0033 | +0.0013 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
