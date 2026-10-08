# Screen scr-zrk-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrm-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7801 | 0.7806 | +0.0006 [-0.0004, +0.0015] | WITHIN | 0.0047 | +0.0007 | +0.0039 |
| squad | in-domain | 11873 | 0.9116 | 0.9130 | +0.0014 [-0.0006, +0.0035] | WITHIN | 0.9054 | +0.0014 | -0.0002 |
| musique | in-domain | 2417 | 0.5600 | 0.5603 | +0.0003 [-0.0052, +0.0060] | WITHIN | 0.4734 | -0.0054 | -0.0083 |
| hotpotqa | zero-shot | 7405 | 0.8457 | 0.8416 | -0.0041 [-0.0075, -0.0009] | WITHIN | 0.6846 | -0.0108 | +0.0026 |
| 2wiki | in-domain | 12576 | 0.8681 | 0.8684 | +0.0004 [-0.0019, +0.0025] | WITHIN | 0.6079 | +0.0024 | +0.0095 |
| webqsp | zero-shot | 1503 | 0.2416 | 0.2432 | +0.0016 [-0.0077, +0.0120] | WITHIN | 0.0535 | +0.0040 | -0.0040 |

## Against outputs\full_zret\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6419 | 0.7806 | +0.1387 [+0.1334, +0.1443] | GAIN | 0.0047 | +0.1446 | +0.2690 |
| squad | in-domain | 11873 | 0.9127 | 0.9130 | +0.0003 [-0.0017, +0.0021] | WITHIN | 0.9054 | +0.0003 | +0.0005 |
| musique | in-domain | 2417 | 0.5623 | 0.5603 | -0.0020 [-0.0080, +0.0042] | WITHIN | 0.4734 | -0.0050 | -0.0087 |
| hotpotqa | zero-shot | 7405 | 0.8445 | 0.8416 | -0.0029 [-0.0066, +0.0008] | WITHIN | 0.6846 | -0.0084 | +0.0063 |
| 2wiki | in-domain | 12576 | 0.8728 | 0.8684 | -0.0044 [-0.0065, -0.0023] | WITHIN | 0.6079 | -0.0107 | +0.0065 |
| webqsp | zero-shot | 1503 | 0.1195 | 0.2432 | +0.1237 [+0.1031, +0.1440] | GAIN | 0.0535 | +0.0991 | +0.0639 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7806 | +0.1267 [+0.1214, +0.1317] | GAIN | 0.0047 | +0.1346 | +0.2595 |
| squad | in-domain | 11873 | 0.9116 | 0.9130 | +0.0013 [-0.0009, +0.0036] | WITHIN | 0.9054 | +0.0013 | +0.0019 |
| musique | in-domain | 2417 | 0.5550 | 0.5603 | +0.0052 [-0.0005, +0.0115] | WITHIN | 0.4734 | +0.0124 | +0.0041 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8416 | -0.0169 [-0.0211, -0.0130] | LOSS | 0.6846 | -0.0343 | +0.0069 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8684 | +0.0012 [-0.0013, +0.0035] | WITHIN | 0.6079 | +0.0052 | +0.0030 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2432 | +0.1232 [+0.1017, +0.1447] | GAIN | 0.0535 | +0.0971 | +0.0592 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
