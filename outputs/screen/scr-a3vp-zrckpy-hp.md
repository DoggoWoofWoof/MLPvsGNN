# Screen scr-a3vp-zrckpy-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1gb-zrckp-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7997 | 0.8026 | +0.0030 [+0.0021, +0.0039] | WITHIN | 0.0047 | +0.0038 | +0.0106 |
| squad | in-domain | 11873 | 0.9122 | 0.9125 | +0.0003 [-0.0016, +0.0022] | WITHIN | 0.9054 | +0.0003 | +0.0013 |
| musique | in-domain | 2417 | 0.5695 | 0.5637 | -0.0058 [-0.0114, -0.0003] | WITHIN | 0.4734 | -0.0062 | -0.0128 |
| hotpotqa | zero-shot | 7405 | 0.8546 | 0.8509 | -0.0037 [-0.0067, -0.0007] | WITHIN | 0.6846 | -0.0070 | -0.0127 |
| 2wiki | in-domain | 12576 | 0.9340 | 0.9286 | -0.0054 [-0.0071, -0.0036] | WITHIN | 0.6079 | -0.0118 | +0.0053 |
| webqsp | zero-shot | 1503 | 0.3364 | 0.3183 | -0.0181 [-0.0295, -0.0069] | LOSS | 0.0535 | -0.0180 | -0.0106 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8026 | +0.1487 [+0.1431, +0.1540] | GAIN | 0.0047 | +0.1536 | +0.2880 |
| squad | in-domain | 11873 | 0.9116 | 0.9125 | +0.0008 [-0.0013, +0.0031] | WITHIN | 0.9054 | +0.0008 | +0.0024 |
| musique | in-domain | 2417 | 0.5550 | 0.5637 | +0.0087 [+0.0003, +0.0168] | GAIN | 0.4734 | +0.0128 | +0.0190 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8509 | -0.0076 [-0.0124, -0.0030] | LOSS | 0.6846 | -0.0172 | -0.0506 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9286 | +0.0614 [+0.0582, +0.0649] | GAIN | 0.6079 | +0.1234 | +0.0691 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3183 | +0.1983 [+0.1741, +0.2217] | GAIN | 0.0535 | +0.1550 | +0.0552 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
