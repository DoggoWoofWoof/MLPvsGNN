# Screen scr-r2ws-zrckpb-s2-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s2-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4147 | 0.4572 | +0.0426 [+0.0365, +0.0486] | GAIN | 0.0047 | +0.0300 | +0.0525 |
| squad | in-domain | 11873 | 0.9116 | 0.9090 | -0.0027 [-0.0049, -0.0005] | WITHIN | 0.9054 | -0.0027 | -0.0059 |
| musique | in-domain | 2417 | 0.5786 | 0.5571 | -0.0215 [-0.0280, -0.0147] | LOSS | 0.4734 | -0.0207 | -0.0360 |
| hotpotqa | zero-shot | 7405 | 0.8583 | 0.8469 | -0.0115 [-0.0147, -0.0082] | LOSS | 0.6846 | -0.0185 | -0.0166 |
| 2wiki | in-domain | 12576 | 0.9313 | 0.9302 | -0.0011 [-0.0028, +0.0006] | WITHIN | 0.6079 | -0.0014 | -0.0083 |
| webqsp | in-domain | 1503 | 0.6393 | 0.6514 | +0.0122 [+0.0001, +0.0239] | GAIN | 0.0535 | +0.0106 | +0.0020 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
