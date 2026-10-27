# Screen scr-kg-base-zg6-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zspk-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8037 | 0.8115 | +0.0078 [+0.0063, +0.0093] | GAIN | 0.0047 | +0.0092 | +0.0275 |
| squad | in-domain | 11873 | 0.9139 | 0.9122 | -0.0017 [-0.0038, +0.0005] | WITHIN | 0.9054 | -0.0017 | +0.0053 |
| musique | in-domain | 2417 | 0.5754 | 0.5953 | +0.0199 [+0.0139, +0.0255] | GAIN | 0.4734 | +0.0327 | +0.0157 |
| hotpotqa | zero-shot | 7405 | 0.8550 | 0.8391 | -0.0159 [-0.0197, -0.0120] | LOSS | 0.6846 | -0.0181 | -0.0330 |
| 2wiki | in-domain | 12576 | 0.9265 | 0.9463 | +0.0198 [+0.0177, +0.0218] | GAIN | 0.6079 | +0.0615 | -0.0110 |
| webqsp | zero-shot | 1503 | 0.2640 | 0.2430 | -0.0210 [-0.0410, -0.0013] | LOSS | 0.0535 | -0.0146 | +0.0086 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
