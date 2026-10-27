# Screen scr-u2n-zg6e (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zspk (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8013 | 0.8130 | +0.0117 [+0.0100, +0.0135] | GAIN | 0.0047 | +0.0135 | +0.0408 |
| squad | in-domain | 11873 | 0.9102 | 0.9133 | +0.0031 [+0.0009, +0.0054] | WITHIN | 0.9054 | +0.0031 | +0.0030 |
| musique | zero-shot | 2417 | 0.5146 | 0.5129 | -0.0017 [-0.0088, +0.0052] | WITHIN | 0.4734 | +0.0066 | -0.0128 |
| hotpotqa | in-domain | 7405 | 0.9086 | 0.9120 | +0.0034 [+0.0009, +0.0060] | WITHIN | 0.6846 | +0.0058 | +0.0228 |
| 2wiki | in-domain | 12576 | 0.9364 | 0.9438 | +0.0074 [+0.0057, +0.0090] | WITHIN | 0.6079 | +0.0223 | +0.0107 |
| webqsp | zero-shot | 1503 | 0.3151 | 0.2163 | -0.0988 [-0.1198, -0.0773] | LOSS | 0.0535 | -0.0758 | -0.0585 |

## Against outputs\screen\fits\scr-kg-base-zg6 (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8127 | 0.8130 | +0.0003 [-0.0005, +0.0011] | WITHIN | 0.0047 | +0.0002 | +0.0011 |
| squad | in-domain | 11873 | 0.9120 | 0.9133 | +0.0013 [-0.0005, +0.0032] | WITHIN | 0.9054 | +0.0013 | +0.0002 |
| musique | zero-shot | 2417 | 0.5138 | 0.5129 | -0.0008 [-0.0054, +0.0041] | WITHIN | 0.4734 | -0.0012 | -0.0248 |
| hotpotqa | in-domain | 7405 | 0.9127 | 0.9120 | -0.0007 [-0.0026, +0.0014] | WITHIN | 0.6846 | -0.0011 | +0.0086 |
| 2wiki | in-domain | 12576 | 0.9448 | 0.9438 | -0.0010 [-0.0023, +0.0003] | WITHIN | 0.6079 | -0.0022 | -0.0020 |
| webqsp | zero-shot | 1503 | 0.2398 | 0.2163 | -0.0235 [-0.0372, -0.0108] | LOSS | 0.0535 | -0.0180 | -0.0166 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
