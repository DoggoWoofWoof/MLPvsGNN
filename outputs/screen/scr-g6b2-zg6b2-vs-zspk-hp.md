# Screen scr-g6b2-zg6b2-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zspk-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8037 | 0.8115 | +0.0079 [+0.0063, +0.0094] | GAIN | 0.0047 | +0.0091 | +0.0289 |
| squad | in-domain | 11873 | 0.9139 | 0.9127 | -0.0012 [-0.0032, +0.0008] | WITHIN | 0.9054 | -0.0012 | +0.0029 |
| musique | in-domain | 2417 | 0.5754 | 0.6002 | +0.0248 [+0.0189, +0.0305] | GAIN | 0.4734 | +0.0381 | +0.0223 |
| hotpotqa | zero-shot | 7405 | 0.8550 | 0.8367 | -0.0182 [-0.0223, -0.0142] | LOSS | 0.6846 | -0.0193 | -0.0452 |
| 2wiki | in-domain | 12576 | 0.9265 | 0.9420 | +0.0154 [+0.0134, +0.0175] | GAIN | 0.6079 | +0.0502 | +0.0066 |
| webqsp | zero-shot | 1503 | 0.2640 | 0.2501 | -0.0139 [-0.0343, +0.0058] | WITHIN | 0.0535 | -0.0013 | +0.0073 |

## Against outputs\screen\fits\scr-kg-base-zg6-hp (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8115 | 0.8115 | +0.0000 [-0.0008, +0.0009] | WITHIN | 0.0047 | -0.0001 | +0.0014 |
| squad | in-domain | 11873 | 0.9122 | 0.9127 | +0.0005 [-0.0013, +0.0024] | WITHIN | 0.9054 | +0.0005 | -0.0024 |
| musique | in-domain | 2417 | 0.5953 | 0.6002 | +0.0049 [-0.0000, +0.0097] | WITHIN | 0.4734 | +0.0054 | +0.0066 |
| hotpotqa | zero-shot | 7405 | 0.8391 | 0.8367 | -0.0024 [-0.0053, +0.0006] | WITHIN | 0.6846 | -0.0012 | -0.0123 |
| 2wiki | in-domain | 12576 | 0.9463 | 0.9420 | -0.0044 [-0.0058, -0.0029] | WITHIN | 0.6079 | -0.0114 | +0.0176 |
| webqsp | zero-shot | 1503 | 0.2430 | 0.2501 | +0.0071 [-0.0066, +0.0210] | WITHIN | 0.0535 | +0.0133 | -0.0013 |

## Against outputs\screen\fits\scr-g6b-zg6b-hp (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8115 | 0.8115 | +0.0001 [-0.0008, +0.0010] | WITHIN | 0.0047 | -0.0003 | +0.0024 |
| squad | in-domain | 11873 | 0.9130 | 0.9127 | -0.0003 [-0.0019, +0.0013] | WITHIN | 0.9054 | -0.0003 | +0.0000 |
| musique | in-domain | 2417 | 0.5934 | 0.6002 | +0.0068 [+0.0020, +0.0115] | WITHIN | 0.4734 | +0.0050 | +0.0074 |
| hotpotqa | zero-shot | 7405 | 0.8388 | 0.8367 | -0.0020 [-0.0048, +0.0009] | WITHIN | 0.6846 | -0.0012 | -0.0073 |
| 2wiki | in-domain | 12576 | 0.9443 | 0.9420 | -0.0023 [-0.0037, -0.0009] | WITHIN | 0.6079 | -0.0065 | +0.0053 |
| webqsp | zero-shot | 1503 | 0.2397 | 0.2501 | +0.0104 [-0.0019, +0.0237] | WITHIN | 0.0535 | +0.0153 | -0.0193 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
