# Screen zrcd-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrm-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7801 | 0.7801 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.0047 | +0.0000 | +0.0000 |
| squad | in-domain | 11873 | 0.9116 | 0.9116 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.9054 | +0.0000 | +0.0000 |
| musique | in-domain | 2417 | 0.5600 | 0.5600 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.4734 | +0.0000 | +0.0000 |
| hotpotqa | zero-shot | 7405 | 0.8457 | 0.8457 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.6846 | +0.0000 | +0.0000 |
| 2wiki | in-domain | 12576 | 0.8681 | 0.8681 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.6079 | +0.0000 | +0.0000 |
| webqsp | zero-shot | 1503 | 0.2416 | 0.2860 | +0.0444 [+0.0313, +0.0577] | GAIN | 0.0535 | +0.0386 | -0.0053 |

## Against outputs\full_zret\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6419 | 0.7801 | +0.1381 [+0.1328, +0.1437] | GAIN | 0.0047 | +0.1439 | +0.2651 |
| squad | in-domain | 11873 | 0.9127 | 0.9116 | -0.0012 [-0.0032, +0.0008] | WITHIN | 0.9054 | -0.0012 | +0.0007 |
| musique | in-domain | 2417 | 0.5623 | 0.5600 | -0.0023 [-0.0087, +0.0034] | WITHIN | 0.4734 | +0.0004 | -0.0004 |
| hotpotqa | zero-shot | 7405 | 0.8445 | 0.8457 | +0.0012 [-0.0024, +0.0049] | WITHIN | 0.6846 | +0.0024 | +0.0038 |
| 2wiki | in-domain | 12576 | 0.8728 | 0.8681 | -0.0047 [-0.0068, -0.0025] | WITHIN | 0.6079 | -0.0130 | -0.0029 |
| webqsp | zero-shot | 1503 | 0.1195 | 0.2860 | +0.1665 [+0.1447, +0.1888] | GAIN | 0.0535 | +0.1337 | +0.0625 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7801 | +0.1261 [+0.1209, +0.1312] | GAIN | 0.0047 | +0.1339 | +0.2556 |
| squad | in-domain | 11873 | 0.9116 | 0.9116 | -0.0001 [-0.0024, +0.0023] | WITHIN | 0.9054 | -0.0001 | +0.0020 |
| musique | in-domain | 2417 | 0.5550 | 0.5600 | +0.0049 [-0.0011, +0.0113] | WITHIN | 0.4734 | +0.0178 | +0.0124 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8457 | -0.0128 [-0.0165, -0.0088] | LOSS | 0.6846 | -0.0235 | +0.0043 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8681 | +0.0008 [-0.0015, +0.0032] | WITHIN | 0.6079 | +0.0028 | -0.0064 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2860 | +0.1660 [+0.1442, +0.1883] | GAIN | 0.0535 | +0.1317 | +0.0579 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
