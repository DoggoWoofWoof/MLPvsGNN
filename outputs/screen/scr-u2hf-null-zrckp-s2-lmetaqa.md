# Screen scr-u2hf-base-zrckp-s2-lmetaqa (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zrckp-lmetaqa (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.2463 | 0.2472 | +0.0009 [-0.0034, +0.0056] | WITHIN | 0.0047 | +0.0024 | +0.0038 |
| squad | in-domain | 11873 | 0.9095 | 0.9120 | +0.0024 [+0.0004, +0.0046] | WITHIN | 0.9054 | +0.0024 | -0.0009 |
| musique | in-domain | 2417 | 0.5757 | 0.5696 | -0.0061 [-0.0115, -0.0011] | WITHIN | 0.4734 | -0.0062 | -0.0174 |
| hotpotqa | in-domain | 7405 | 0.9063 | 0.9030 | -0.0034 [-0.0056, -0.0011] | WITHIN | 0.6846 | -0.0062 | +0.0027 |
| 2wiki | in-domain | 12576 | 0.9286 | 0.9312 | +0.0026 [+0.0010, +0.0043] | WITHIN | 0.6079 | +0.0062 | +0.0026 |
| webqsp | zero-shot | 1503 | 0.1152 | 0.1095 | -0.0058 [-0.0138, +0.0020] | WITHIN | 0.0535 | -0.0047 | -0.0060 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
