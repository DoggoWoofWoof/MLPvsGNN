# Screen scr-u1f-zsp-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zspg-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7807 | 0.7873 | +0.0065 [+0.0046, +0.0085] | WITHIN | 0.0047 | +0.0073 | -0.0026 |
| squad | in-domain | 11873 | 0.9109 | 0.9135 | +0.0026 [+0.0006, +0.0048] | WITHIN | 0.9054 | +0.0026 | +0.0006 |
| musique | in-domain | 2417 | 0.5647 | 0.5685 | +0.0038 [-0.0023, +0.0099] | WITHIN | 0.4734 | +0.0058 | -0.0099 |
| hotpotqa | zero-shot | 7405 | 0.8435 | 0.8082 | -0.0353 [-0.0401, -0.0308] | LOSS | 0.6846 | -0.0678 | -0.0012 |
| 2wiki | in-domain | 12576 | 0.8744 | 0.8617 | -0.0127 [-0.0161, -0.0092] | LOSS | 0.6079 | -0.0328 | -0.0024 |
| webqsp | zero-shot | 1503 | 0.2262 | 0.2414 | +0.0152 [-0.0002, +0.0301] | WITHIN | 0.0535 | +0.0106 | -0.0126 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7873 | +0.1333 [+0.1278, +0.1386] | GAIN | 0.0047 | +0.1413 | +0.2537 |
| squad | in-domain | 11873 | 0.9116 | 0.9135 | +0.0019 [-0.0005, +0.0042] | WITHIN | 0.9054 | +0.0019 | +0.0024 |
| musique | in-domain | 2417 | 0.5550 | 0.5685 | +0.0135 [+0.0071, +0.0206] | GAIN | 0.4734 | +0.0244 | -0.0037 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8082 | -0.0504 [-0.0554, -0.0455] | LOSS | 0.6846 | -0.0963 | +0.0026 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8617 | -0.0055 [-0.0088, -0.0020] | WITHIN | 0.6079 | -0.0109 | +0.0009 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2414 | +0.1214 [+0.0983, +0.1438] | GAIN | 0.0535 | +0.0918 | +0.0512 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
