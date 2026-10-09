# Screen scr-zgr (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrm (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7789 | 0.7823 | +0.0034 [+0.0020, +0.0047] | WITHIN | 0.0047 | +0.0021 | +0.0064 |
| squad | in-domain | 11873 | 0.9111 | 0.8993 | -0.0118 [-0.0149, -0.0086] | LOSS | 0.9054 | -0.0118 | -0.0280 |
| musique | zero-shot | 2417 | 0.5231 | 0.5174 | -0.0057 [-0.0122, +0.0007] | WITHIN | 0.4734 | -0.0017 | -0.0943 |
| hotpotqa | in-domain | 7405 | 0.9040 | 0.8977 | -0.0063 [-0.0091, -0.0033] | WITHIN | 0.6846 | -0.0101 | -0.0147 |
| 2wiki | in-domain | 12576 | 0.8709 | 0.8706 | -0.0003 [-0.0024, +0.0018] | WITHIN | 0.6079 | +0.0015 | -0.0253 |
| webqsp | zero-shot | 1503 | 0.3128 | 0.3124 | -0.0004 [-0.0133, +0.0116] | WITHIN | 0.0535 | -0.0047 | +0.0146 |

## Against outputs\screen\fits\scr-zret (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6496 | 0.7823 | +0.1327 [+0.1273, +0.1378] | GAIN | 0.0047 | +0.1350 | +0.2597 |
| squad | in-domain | 11873 | 0.9106 | 0.8993 | -0.0114 [-0.0145, -0.0083] | LOSS | 0.9054 | -0.0114 | -0.0257 |
| musique | zero-shot | 2417 | 0.3897 | 0.5174 | +0.1277 [+0.1175, +0.1382] | GAIN | 0.4734 | +0.0799 | +0.1361 |
| hotpotqa | in-domain | 7405 | 0.9061 | 0.8977 | -0.0084 [-0.0115, -0.0053] | LOSS | 0.6846 | -0.0147 | -0.0193 |
| 2wiki | in-domain | 12576 | 0.8714 | 0.8706 | -0.0009 [-0.0030, +0.0013] | WITHIN | 0.6079 | +0.0004 | -0.0316 |
| webqsp | zero-shot | 1503 | 0.1766 | 0.3124 | +0.1358 [+0.1130, +0.1595] | GAIN | 0.0535 | +0.1045 | +0.1257 |

## Against outputs\step1\fits\L-musique (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7823 | +0.1286 [+0.1234, +0.1337] | GAIN | 0.0047 | +0.1295 | +0.2586 |
| squad | in-domain | 11873 | 0.9100 | 0.8993 | -0.0107 [-0.0138, -0.0076] | LOSS | 0.9054 | -0.0107 | -0.0296 |
| musique | zero-shot | 2417 | 0.2696 | 0.5174 | +0.2479 [+0.2362, +0.2593] | GAIN | 0.4734 | +0.1481 | +0.2900 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.8977 | -0.0038 [-0.0068, -0.0008] | WITHIN | 0.6846 | -0.0072 | -0.0159 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8706 | -0.0026 [-0.0048, -0.0005] | WITHIN | 0.6079 | -0.0051 | -0.0274 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3124 | +0.1346 [+0.1125, +0.1563] | GAIN | 0.0535 | +0.1051 | +0.1204 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
