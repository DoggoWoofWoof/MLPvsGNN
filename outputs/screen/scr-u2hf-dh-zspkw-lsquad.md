# Screen scr-u2hf-zspkw-lsquad (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zspk-lsquad (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8019 | 0.8023 | +0.0004 [-0.0004, +0.0013] | WITHIN | 0.0047 | +0.0000 | +0.0009 |
| squad | zero-shot | 11873 | 0.8929 | 0.8948 | +0.0019 [-0.0004, +0.0043] | WITHIN | 0.9054 | +0.0019 | +0.0054 |
| musique | in-domain | 2417 | 0.5807 | 0.5687 | -0.0120 [-0.0172, -0.0068] | LOSS | 0.4734 | -0.0120 | -0.0037 |
| hotpotqa | in-domain | 7405 | 0.9092 | 0.9108 | +0.0016 [-0.0005, +0.0037] | WITHIN | 0.6846 | +0.0020 | +0.0003 |
| 2wiki | in-domain | 12576 | 0.9385 | 0.9297 | -0.0088 [-0.0104, -0.0073] | LOSS | 0.6079 | -0.0243 | -0.0016 |
| webqsp | zero-shot | 1503 | 0.2957 | 0.3018 | +0.0061 [-0.0021, +0.0144] | WITHIN | 0.0535 | +0.0053 | +0.0007 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
