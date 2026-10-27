# Screen scr-r2t-zrckpb (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zrckp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7995 | 0.8050 | +0.0054 [+0.0042, +0.0067] | WITHIN | 0.0047 | +0.0052 | +0.0232 |
| squad | in-domain | 11873 | 0.9116 | 0.9106 | -0.0010 [-0.0031, +0.0010] | WITHIN | 0.9054 | -0.0010 | -0.0007 |
| musique | zero-shot | 2417 | 0.5129 | 0.5221 | +0.0091 [+0.0035, +0.0149] | GAIN | 0.4734 | +0.0103 | -0.0261 |
| hotpotqa | in-domain | 7405 | 0.9041 | 0.9063 | +0.0022 [-0.0003, +0.0048] | WITHIN | 0.6846 | +0.0054 | -0.0161 |
| 2wiki | in-domain | 12576 | 0.9319 | 0.9346 | +0.0026 [+0.0011, +0.0041] | WITHIN | 0.6079 | +0.0078 | -0.0093 |
| webqsp | zero-shot | 1503 | 0.3570 | 0.3689 | +0.0120 [+0.0014, +0.0227] | GAIN | 0.0535 | +0.0080 | +0.0333 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
