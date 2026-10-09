# Screen scr-zrg (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrm (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7789 | 0.7801 | +0.0012 [+0.0004, +0.0022] | WITHIN | 0.0047 | +0.0010 | -0.0002 |
| squad | in-domain | 11873 | 0.9111 | 0.9127 | +0.0016 [-0.0004, +0.0037] | WITHIN | 0.9054 | +0.0016 | -0.0013 |
| musique | zero-shot | 2417 | 0.5231 | 0.5191 | -0.0040 [-0.0092, +0.0011] | WITHIN | 0.4734 | -0.0025 | -0.0285 |
| hotpotqa | in-domain | 7405 | 0.9040 | 0.9061 | +0.0022 [-0.0001, +0.0045] | WITHIN | 0.6846 | +0.0036 | -0.0016 |
| 2wiki | in-domain | 12576 | 0.8709 | 0.8736 | +0.0027 [+0.0010, +0.0046] | WITHIN | 0.6079 | +0.0066 | -0.0062 |
| webqsp | zero-shot | 1503 | 0.3128 | 0.3146 | +0.0018 [-0.0077, +0.0109] | WITHIN | 0.0535 | +0.0020 | -0.0060 |

## Against outputs\screen\fits\scr-zgs (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6527 | 0.7801 | +0.1275 [+0.1223, +0.1327] | GAIN | 0.0047 | +0.1323 | +0.2507 |
| squad | in-domain | 11873 | 0.9116 | 0.9127 | +0.0011 [-0.0008, +0.0030] | WITHIN | 0.9054 | +0.0011 | -0.0019 |
| musique | zero-shot | 2417 | 0.3798 | 0.5191 | +0.1393 [+0.1296, +0.1493] | GAIN | 0.4734 | +0.0902 | +0.2185 |
| hotpotqa | in-domain | 7405 | 0.9047 | 0.9061 | +0.0014 [-0.0009, +0.0038] | WITHIN | 0.6846 | +0.0028 | -0.0146 |
| 2wiki | in-domain | 12576 | 0.8719 | 0.8736 | +0.0017 [-0.0002, +0.0036] | WITHIN | 0.6079 | +0.0007 | -0.0106 |
| webqsp | zero-shot | 1503 | 0.1875 | 0.3146 | +0.1271 [+0.1048, +0.1500] | GAIN | 0.0535 | +0.1018 | +0.1045 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7801 | +0.1264 [+0.1213, +0.1318] | GAIN | 0.0047 | +0.1284 | +0.2519 |
| squad | in-domain | 11873 | 0.9100 | 0.9127 | +0.0027 [+0.0005, +0.0050] | WITHIN | 0.9054 | +0.0027 | -0.0029 |
| musique | zero-shot | 2417 | 0.2696 | 0.5191 | +0.2495 [+0.2384, +0.2612] | GAIN | 0.4734 | +0.1473 | +0.3558 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9061 | +0.0047 [+0.0021, +0.0072] | WITHIN | 0.6846 | +0.0066 | -0.0028 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8736 | +0.0004 [-0.0015, +0.0024] | WITHIN | 0.6079 | +0.0000 | -0.0083 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3146 | +0.1367 [+0.1146, +0.1590] | GAIN | 0.0535 | +0.1118 | +0.0998 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
