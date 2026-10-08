# Screen scr-nullx-L-squad-s2 (p@swa, s1eval carves)

## Against outputs\step1\fits\L-squad (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6535 | 0.6525 | -0.0010 [-0.0039, +0.0020] | WITHIN | 0.0047 | -0.0045 | -0.0018 |
| squad | zero-shot | 11873 | 0.8797 | 0.8818 | +0.0021 [-0.0006, +0.0050] | WITHIN | 0.9054 | +0.0021 | +0.0129 |
| musique | in-domain | 2417 | 0.5743 | 0.5599 | -0.0144 [-0.0200, -0.0089] | LOSS | 0.4734 | -0.0145 | -0.0087 |
| hotpotqa | in-domain | 7405 | 0.9045 | 0.9012 | -0.0032 [-0.0059, -0.0008] | WITHIN | 0.6846 | -0.0045 | +0.0157 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.8641 | -0.0083 [-0.0104, -0.0063] | LOSS | 0.6079 | -0.0107 | +0.0028 |
| webqsp | zero-shot | 1503 | 0.1302 | 0.1474 | +0.0171 [+0.0086, +0.0261] | GAIN | 0.0535 | +0.0126 | +0.0027 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
