# Screen scr-zgf-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrm-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7801 | 0.7815 | +0.0015 [+0.0002, +0.0028] | WITHIN | 0.0047 | -0.0014 | +0.0081 |
| squad | in-domain | 11873 | 0.9116 | 0.8972 | -0.0143 [-0.0175, -0.0112] | LOSS | 0.9054 | -0.0143 | -0.0286 |
| musique | in-domain | 2417 | 0.5600 | 0.5001 | -0.0599 [-0.0683, -0.0511] | LOSS | 0.4734 | -0.0662 | -0.1361 |
| hotpotqa | zero-shot | 7405 | 0.8457 | 0.8020 | -0.0437 [-0.0481, -0.0392] | LOSS | 0.6846 | -0.0810 | -0.0570 |
| 2wiki | in-domain | 12576 | 0.8681 | 0.8626 | -0.0055 [-0.0081, -0.0028] | WITHIN | 0.6079 | -0.0095 | -0.0358 |
| webqsp | zero-shot | 1503 | 0.2416 | 0.2407 | -0.0008 [-0.0122, +0.0108] | WITHIN | 0.0535 | -0.0033 | +0.0040 |

## Against outputs\full_zret\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6419 | 0.7815 | +0.1396 [+0.1343, +0.1452] | GAIN | 0.0047 | +0.1425 | +0.2732 |
| squad | in-domain | 11873 | 0.9127 | 0.8972 | -0.0155 [-0.0188, -0.0123] | LOSS | 0.9054 | -0.0155 | -0.0279 |
| musique | in-domain | 2417 | 0.5623 | 0.5001 | -0.0622 [-0.0709, -0.0539] | LOSS | 0.4734 | -0.0658 | -0.1365 |
| hotpotqa | zero-shot | 7405 | 0.8445 | 0.8020 | -0.0425 [-0.0471, -0.0377] | LOSS | 0.6846 | -0.0786 | -0.0532 |
| 2wiki | in-domain | 12576 | 0.8728 | 0.8626 | -0.0102 [-0.0127, -0.0076] | LOSS | 0.6079 | -0.0225 | -0.0387 |
| webqsp | zero-shot | 1503 | 0.1195 | 0.2407 | +0.1212 [+0.1005, +0.1413] | GAIN | 0.0535 | +0.0918 | +0.0719 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7815 | +0.1276 [+0.1225, +0.1327] | GAIN | 0.0047 | +0.1324 | +0.2637 |
| squad | in-domain | 11873 | 0.9116 | 0.8972 | -0.0144 [-0.0176, -0.0111] | LOSS | 0.9054 | -0.0144 | -0.0265 |
| musique | in-domain | 2417 | 0.5550 | 0.5001 | -0.0550 [-0.0631, -0.0460] | LOSS | 0.4734 | -0.0484 | -0.1237 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8020 | -0.0565 [-0.0612, -0.0517] | LOSS | 0.6846 | -0.1045 | -0.0527 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8626 | -0.0047 [-0.0072, -0.0019] | WITHIN | 0.6079 | -0.0067 | -0.0422 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2407 | +0.1208 [+0.1000, +0.1422] | GAIN | 0.0535 | +0.0898 | +0.0672 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
