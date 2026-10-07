# Screen scr-ztop50-hp (p@swa, s1eval carves)

## Against outputs\step1\fits\L-hotpotqa (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.6287 | -0.0252 [-0.0291, -0.0214] | LOSS | 0.0047 | -0.0202 | -0.0262 |
| squad | in-domain | 11873 | 0.9116 | 0.9113 | -0.0003 [-0.0029, +0.0020] | WITHIN | 0.9054 | -0.0003 | +0.0012 |
| musique | in-domain | 2417 | 0.5550 | 0.5665 | +0.0115 [+0.0056, +0.0178] | GAIN | 0.4734 | +0.0236 | +0.0132 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8504 | -0.0082 [-0.0117, -0.0042] | LOSS | 0.6846 | -0.0161 | -0.0007 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8647 | -0.0025 [-0.0048, -0.0001] | WITHIN | 0.6079 | -0.0041 | +0.0002 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.1028 | -0.0172 [-0.0273, -0.0067] | LOSS | 0.0535 | -0.0140 | -0.0040 |

## Against outputs\full_zret\fits\L-hotpotqa (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6419 | 0.6287 | -0.0132 [-0.0169, -0.0096] | LOSS | 0.0047 | -0.0102 | -0.0167 |
| squad | in-domain | 11873 | 0.9127 | 0.9113 | -0.0014 [-0.0035, +0.0007] | WITHIN | 0.9054 | -0.0014 | -0.0002 |
| musique | in-domain | 2417 | 0.5623 | 0.5665 | +0.0042 [-0.0023, +0.0105] | WITHIN | 0.4734 | +0.0062 | +0.0004 |
| hotpotqa | zero-shot | 7405 | 0.8445 | 0.8504 | +0.0059 [+0.0021, +0.0098] | WITHIN | 0.6846 | +0.0099 | -0.0012 |
| 2wiki | in-domain | 12576 | 0.8728 | 0.8647 | -0.0081 [-0.0103, -0.0058] | LOSS | 0.6079 | -0.0200 | +0.0037 |
| webqsp | zero-shot | 1503 | 0.1195 | 0.1028 | -0.0167 [-0.0268, -0.0066] | LOSS | 0.0535 | -0.0120 | +0.0007 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
