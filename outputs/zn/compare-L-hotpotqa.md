# Screen scr-zn-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrct-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7793 | 0.2192 | -0.5600 [-0.5687, -0.5515] | LOSS | 0.0047 | -0.5192 | -0.7023 |
| squad | in-domain | 11873 | 0.9112 | 0.9093 | -0.0019 [-0.0045, +0.0007] | WITHIN | 0.9054 | -0.0019 | -0.0022 |
| musique | in-domain | 2417 | 0.5645 | 0.5184 | -0.0461 [-0.0546, -0.0380] | LOSS | 0.4734 | -0.0732 | +0.0103 |
| hotpotqa | zero-shot | 7405 | 0.8451 | 0.7398 | -0.1053 [-0.1111, -0.0997] | LOSS | 0.6846 | -0.2042 | +0.0001 |
| 2wiki | in-domain | 12576 | 0.8731 | 0.7970 | -0.0762 [-0.0799, -0.0725] | LOSS | 0.6079 | -0.1533 | +0.0058 |
| webqsp | zero-shot | 1503 | 0.2865 | 0.0973 | -0.1892 [-0.2117, -0.1668] | LOSS | 0.0535 | -0.1564 | -0.0419 |

## Against outputs\screen\fits\scr-zspg-hp (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7807 | 0.2192 | -0.5615 [-0.5700, -0.5529] | LOSS | 0.0047 | -0.5198 | -0.7032 |
| squad | in-domain | 11873 | 0.9109 | 0.9093 | -0.0016 [-0.0041, +0.0009] | WITHIN | 0.9054 | -0.0016 | -0.0009 |
| musique | in-domain | 2417 | 0.5647 | 0.5184 | -0.0463 [-0.0546, -0.0386] | LOSS | 0.4734 | -0.0761 | +0.0046 |
| hotpotqa | zero-shot | 7405 | 0.8435 | 0.7398 | -0.1036 [-0.1095, -0.0982] | LOSS | 0.6846 | -0.2000 | -0.0036 |
| 2wiki | in-domain | 12576 | 0.8744 | 0.7970 | -0.0775 [-0.0813, -0.0738] | LOSS | 0.6079 | -0.1578 | -0.0063 |
| webqsp | zero-shot | 1503 | 0.2262 | 0.0973 | -0.1289 [-0.1503, -0.1078] | LOSS | 0.0535 | -0.1011 | -0.0585 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
