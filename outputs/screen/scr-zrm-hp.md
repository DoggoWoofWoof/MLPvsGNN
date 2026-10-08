# Screen scr-zrm-hp (p@swa, s1eval carves)

## Against outputs\full_zret\fits\L-hotpotqa (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6419 | 0.7801 | +0.1381 [+0.1328, +0.1437] | GAIN | 0.0047 | +0.1439 | +0.2651 |
| squad | in-domain | 11873 | 0.9127 | 0.9116 | -0.0012 [-0.0032, +0.0008] | WITHIN | 0.9054 | -0.0012 | +0.0007 |
| musique | in-domain | 2417 | 0.5623 | 0.5600 | -0.0023 [-0.0087, +0.0034] | WITHIN | 0.4734 | +0.0004 | -0.0004 |
| hotpotqa | zero-shot | 7405 | 0.8445 | 0.8457 | +0.0012 [-0.0024, +0.0049] | WITHIN | 0.6846 | +0.0024 | +0.0038 |
| 2wiki | in-domain | 12576 | 0.8728 | 0.8681 | -0.0047 [-0.0068, -0.0025] | WITHIN | 0.6079 | -0.0130 | -0.0029 |
| webqsp | zero-shot | 1503 | 0.1195 | 0.2416 | +0.1221 [+0.1018, +0.1424] | GAIN | 0.0535 | +0.0951 | +0.0679 |

## Against outputs\screen\fits\scr-rmatch-hp (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7790 | 0.7801 | +0.0011 [+0.0000, +0.0022] | WITHIN | 0.0047 | +0.0013 | +0.0031 |
| squad | in-domain | 11873 | 0.9101 | 0.9116 | +0.0014 [-0.0006, +0.0035] | WITHIN | 0.9054 | +0.0014 | +0.0026 |
| musique | in-domain | 2417 | 0.5588 | 0.5600 | +0.0012 [-0.0050, +0.0071] | WITHIN | 0.4734 | +0.0074 | +0.0087 |
| hotpotqa | zero-shot | 7405 | 0.8513 | 0.8457 | -0.0056 [-0.0092, -0.0018] | WITHIN | 0.6846 | -0.0103 | +0.0070 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.8681 | -0.0043 [-0.0066, -0.0021] | WITHIN | 0.6079 | -0.0122 | -0.0052 |
| webqsp | zero-shot | 1503 | 0.2646 | 0.2416 | -0.0230 [-0.0349, -0.0113] | LOSS | 0.0535 | -0.0193 | -0.0073 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7801 | +0.1261 [+0.1209, +0.1312] | GAIN | 0.0047 | +0.1339 | +0.2556 |
| squad | in-domain | 11873 | 0.9116 | 0.9116 | -0.0001 [-0.0024, +0.0023] | WITHIN | 0.9054 | -0.0001 | +0.0020 |
| musique | in-domain | 2417 | 0.5550 | 0.5600 | +0.0049 [-0.0011, +0.0113] | WITHIN | 0.4734 | +0.0178 | +0.0124 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8457 | -0.0128 [-0.0165, -0.0088] | LOSS | 0.6846 | -0.0235 | +0.0043 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8681 | +0.0008 [-0.0015, +0.0032] | WITHIN | 0.6079 | +0.0028 | -0.0064 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2416 | +0.1216 [+0.1004, +0.1424] | GAIN | 0.0535 | +0.0931 | +0.0632 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
