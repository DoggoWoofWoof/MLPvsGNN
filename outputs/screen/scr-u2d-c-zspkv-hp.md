# Screen scr-u2d-new-zspkv-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zspk-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8043 | 0.7990 | -0.0053 [-0.0069, -0.0037] | WITHIN | 0.0047 | -0.0056 | -0.0180 |
| squad | in-domain | 11873 | 0.9117 | 0.9143 | +0.0026 [+0.0003, +0.0049] | WITHIN | 0.9054 | +0.0026 | +0.0014 |
| musique | in-domain | 2417 | 0.5800 | 0.5727 | -0.0072 [-0.0132, -0.0006] | WITHIN | 0.4734 | -0.0083 | -0.0128 |
| hotpotqa | zero-shot | 7405 | 0.8512 | 0.8086 | -0.0427 [-0.0473, -0.0384] | LOSS | 0.6846 | -0.0840 | +0.0117 |
| 2wiki | in-domain | 12576 | 0.9290 | 0.9107 | -0.0183 [-0.0213, -0.0152] | LOSS | 0.6079 | -0.0461 | +0.0045 |
| webqsp | zero-shot | 1503 | 0.2773 | 0.2867 | +0.0094 [-0.0042, +0.0230] | WITHIN | 0.0535 | +0.0080 | +0.0186 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
