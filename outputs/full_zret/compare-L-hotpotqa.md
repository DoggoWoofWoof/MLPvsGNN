# Screen L-hotpotqa (p@swa, s1eval carves)

## Against outputs\step1\fits\L-hotpotqa (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.6419 | -0.0121 [-0.0157, -0.0085] | LOSS | 0.0047 | -0.0100 | -0.0095 |
| squad | in-domain | 11873 | 0.9116 | 0.9127 | +0.0011 [-0.0011, +0.0033] | WITHIN | 0.9054 | +0.0011 | +0.0013 |
| musique | in-domain | 2417 | 0.5550 | 0.5623 | +0.0073 [+0.0009, +0.0134] | WITHIN | 0.4734 | +0.0174 | +0.0128 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8445 | -0.0140 [-0.0180, -0.0101] | LOSS | 0.6846 | -0.0259 | +0.0005 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8728 | +0.0056 [+0.0034, +0.0079] | WITHIN | 0.6079 | +0.0158 | -0.0035 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.1195 | -0.0005 [-0.0107, +0.0098] | WITHIN | 0.0535 | -0.0020 | -0.0047 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
