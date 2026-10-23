# Screen scr-u2hf-zspkw-j5 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zspk-j5 (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8007 | 0.8015 | +0.0007 [-0.0000, +0.0016] | WITHIN | 0.0047 | +0.0002 | +0.0037 |
| squad | in-domain | 11873 | 0.9113 | 0.9106 | -0.0008 [-0.0026, +0.0011] | WITHIN | 0.9054 | -0.0008 | +0.0006 |
| musique | in-domain | 2417 | 0.5772 | 0.5831 | +0.0059 [+0.0007, +0.0111] | WITHIN | 0.4734 | +0.0079 | +0.0000 |
| hotpotqa | in-domain | 7405 | 0.9045 | 0.9067 | +0.0022 [-0.0001, +0.0043] | WITHIN | 0.6846 | +0.0032 | +0.0035 |
| 2wiki | in-domain | 12576 | 0.9358 | 0.9378 | +0.0020 [+0.0006, +0.0034] | WITHIN | 0.6079 | +0.0070 | -0.0052 |
| webqsp | zero-shot | 1503 | 0.2894 | 0.2797 | -0.0096 [-0.0178, -0.0015] | LOSS | 0.0535 | -0.0106 | -0.0073 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
