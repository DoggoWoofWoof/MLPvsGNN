# Screen scr-u2g-y-zspk (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zspk (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8021 | 0.8023 | +0.0002 [-0.0008, +0.0012] | WITHIN | 0.0047 | +0.0000 | +0.0019 |
| squad | in-domain | 11873 | 0.9118 | 0.9110 | -0.0008 [-0.0031, +0.0015] | WITHIN | 0.9054 | -0.0008 | +0.0061 |
| musique | zero-shot | 2417 | 0.5143 | 0.5286 | +0.0142 [+0.0074, +0.0212] | GAIN | 0.4734 | +0.0037 | +0.0174 |
| hotpotqa | in-domain | 7405 | 0.9088 | 0.8394 | -0.0694 [-0.0742, -0.0647] | LOSS | 0.6846 | -0.1368 | -0.0134 |
| 2wiki | in-domain | 12576 | 0.9346 | 0.9072 | -0.0274 [-0.0304, -0.0245] | LOSS | 0.6079 | -0.0736 | +0.0072 |
| webqsp | zero-shot | 1503 | 0.3162 | 0.3009 | -0.0154 [-0.0252, -0.0065] | LOSS | 0.0535 | -0.0106 | -0.0033 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
