# Screen scr-u2hf-base-zspk-s2-lsquad (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zspk-lsquad (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8019 | 0.8043 | +0.0024 [+0.0014, +0.0035] | WITHIN | 0.0047 | +0.0017 | +0.0110 |
| squad | zero-shot | 11873 | 0.8929 | 0.8915 | -0.0013 [-0.0039, +0.0012] | WITHIN | 0.9054 | -0.0013 | -0.0013 |
| musique | in-domain | 2417 | 0.5807 | 0.5841 | +0.0034 [-0.0018, +0.0091] | WITHIN | 0.4734 | +0.0070 | -0.0165 |
| hotpotqa | in-domain | 7405 | 0.9092 | 0.9088 | -0.0004 [-0.0026, +0.0019] | WITHIN | 0.6846 | -0.0011 | -0.0068 |
| 2wiki | in-domain | 12576 | 0.9385 | 0.9362 | -0.0023 [-0.0038, -0.0007] | WITHIN | 0.6079 | -0.0042 | -0.0011 |
| webqsp | zero-shot | 1503 | 0.2957 | 0.2919 | -0.0038 [-0.0133, +0.0058] | WITHIN | 0.0535 | -0.0033 | +0.0067 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
