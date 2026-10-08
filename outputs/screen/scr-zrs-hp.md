# Screen scr-zrs-hp (p@swa, s1eval carves)

## Against outputs\full_zret\fits\L-hotpotqa (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6419 | 0.7642 | +0.1223 [+0.1174, +0.1274] | GAIN | 0.0047 | +0.1228 | +0.2161 |
| squad | in-domain | 11873 | 0.9127 | 0.9127 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.9054 | +0.0000 | +0.0000 |
| musique | in-domain | 2417 | 0.5623 | 0.5623 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.4734 | +0.0000 | +0.0000 |
| hotpotqa | zero-shot | 7405 | 0.8445 | 0.8445 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.6846 | +0.0000 | +0.0000 |
| 2wiki | in-domain | 12576 | 0.8728 | 0.8728 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.6079 | +0.0000 | +0.0000 |
| webqsp | zero-shot | 1503 | 0.1195 | 0.2100 | +0.0905 [+0.0734, +0.1083] | GAIN | 0.0535 | +0.0719 | +0.0519 |

## Against outputs\screen\fits\scr-rmatch-hp (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7790 | 0.7642 | -0.0148 [-0.0169, -0.0126] | LOSS | 0.0047 | -0.0197 | -0.0459 |
| squad | in-domain | 11873 | 0.9101 | 0.9127 | +0.0026 [+0.0004, +0.0046] | WITHIN | 0.9054 | +0.0026 | +0.0019 |
| musique | in-domain | 2417 | 0.5588 | 0.5623 | +0.0035 [-0.0026, +0.0098] | WITHIN | 0.4734 | +0.0070 | +0.0091 |
| hotpotqa | zero-shot | 7405 | 0.8513 | 0.8445 | -0.0068 [-0.0107, -0.0030] | WITHIN | 0.6846 | -0.0127 | +0.0032 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.8728 | +0.0004 [-0.0018, +0.0027] | WITHIN | 0.6079 | +0.0008 | -0.0023 |
| webqsp | zero-shot | 1503 | 0.2646 | 0.2100 | -0.0546 [-0.0719, -0.0380] | LOSS | 0.0535 | -0.0426 | -0.0233 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7642 | +0.1102 [+0.1052, +0.1150] | GAIN | 0.0047 | +0.1128 | +0.2066 |
| squad | in-domain | 11873 | 0.9116 | 0.9127 | +0.0011 [-0.0011, +0.0033] | WITHIN | 0.9054 | +0.0011 | +0.0013 |
| musique | in-domain | 2417 | 0.5550 | 0.5623 | +0.0073 [+0.0009, +0.0134] | WITHIN | 0.4734 | +0.0174 | +0.0128 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8445 | -0.0140 [-0.0180, -0.0101] | LOSS | 0.6846 | -0.0259 | +0.0005 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8728 | +0.0056 [+0.0034, +0.0079] | WITHIN | 0.6079 | +0.0158 | -0.0035 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2100 | +0.0900 [+0.0716, +0.1080] | GAIN | 0.0535 | +0.0699 | +0.0472 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
