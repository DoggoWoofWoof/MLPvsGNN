# Screen L-2wiki (p@swa, s1eval carves)

## Against outputs\full_zrct\fits\L-2wiki (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7793 | 0.7795 | +0.0002 [-0.0009, +0.0014] | WITHIN | 0.0047 | +0.0005 | -0.0011 |
| squad | in-domain | 11873 | 0.9121 | 0.9116 | -0.0004 [-0.0024, +0.0016] | WITHIN | 0.9054 | -0.0004 | +0.0019 |
| musique | in-domain | 2417 | 0.5669 | 0.5687 | +0.0018 [-0.0032, +0.0066] | WITHIN | 0.4734 | -0.0041 | -0.0087 |
| hotpotqa | in-domain | 7405 | 0.9001 | 0.9014 | +0.0014 [-0.0010, +0.0035] | WITHIN | 0.6846 | +0.0016 | -0.0093 |
| 2wiki | zero-shot | 12576 | 0.7937 | 0.7963 | +0.0026 [+0.0005, +0.0048] | WITHIN | 0.6079 | +0.0033 | -0.0107 |
| webqsp | zero-shot | 1503 | 0.3042 | 0.2972 | -0.0070 [-0.0242, +0.0091] | WITHIN | 0.0535 | -0.0067 | +0.0499 |

## Against outputs\full_zrm\fits\L-2wiki (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7794 | 0.7795 | +0.0001 [-0.0010, +0.0013] | WITHIN | 0.0047 | +0.0006 | -0.0029 |
| squad | in-domain | 11873 | 0.9093 | 0.9116 | +0.0024 [+0.0004, +0.0044] | WITHIN | 0.9054 | +0.0024 | -0.0003 |
| musique | in-domain | 2417 | 0.5684 | 0.5687 | +0.0003 [-0.0044, +0.0050] | WITHIN | 0.4734 | -0.0074 | -0.0025 |
| hotpotqa | in-domain | 7405 | 0.9034 | 0.9014 | -0.0020 [-0.0044, +0.0002] | WITHIN | 0.6846 | -0.0042 | -0.0045 |
| 2wiki | zero-shot | 12576 | 0.7972 | 0.7963 | -0.0010 [-0.0029, +0.0012] | WITHIN | 0.6079 | -0.0056 | -0.0060 |
| webqsp | zero-shot | 1503 | 0.2644 | 0.2972 | +0.0328 [+0.0190, +0.0469] | GAIN | 0.0535 | +0.0233 | +0.0286 |

## Against outputs\step1\fits\L-2wiki (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6468 | 0.7795 | +0.1327 [+0.1273, +0.1379] | GAIN | 0.0047 | +0.1385 | +0.2596 |
| squad | in-domain | 11873 | 0.9112 | 0.9116 | +0.0004 [-0.0017, +0.0025] | WITHIN | 0.9054 | +0.0004 | -0.0007 |
| musique | in-domain | 2417 | 0.5601 | 0.5687 | +0.0086 [+0.0018, +0.0148] | GAIN | 0.4734 | +0.0079 | +0.0012 |
| hotpotqa | in-domain | 7405 | 0.8977 | 0.9014 | +0.0037 [+0.0009, +0.0063] | WITHIN | 0.6846 | +0.0059 | +0.0012 |
| 2wiki | zero-shot | 12576 | 0.8074 | 0.7963 | -0.0111 [-0.0136, -0.0085] | LOSS | 0.6079 | -0.0220 | -0.0009 |
| webqsp | zero-shot | 1503 | 0.1075 | 0.2972 | +0.1897 [+0.1660, +0.2139] | GAIN | 0.0535 | +0.1497 | +0.1118 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
