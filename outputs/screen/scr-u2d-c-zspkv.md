# Screen scr-u2d-new-zspkv (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zspk (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8021 | 0.7997 | -0.0024 [-0.0040, -0.0010] | WITHIN | 0.0047 | -0.0022 | -0.0146 |
| squad | in-domain | 11873 | 0.9118 | 0.9127 | +0.0009 [-0.0014, +0.0033] | WITHIN | 0.9054 | +0.0009 | +0.0064 |
| musique | zero-shot | 2417 | 0.5143 | 0.5264 | +0.0121 [+0.0049, +0.0194] | GAIN | 0.4734 | +0.0124 | +0.0112 |
| hotpotqa | in-domain | 7405 | 0.9088 | 0.8464 | -0.0625 [-0.0671, -0.0577] | LOSS | 0.6846 | -0.1237 | -0.0226 |
| 2wiki | in-domain | 12576 | 0.9346 | 0.9159 | -0.0187 [-0.0216, -0.0157] | LOSS | 0.6079 | -0.0549 | +0.0026 |
| webqsp | zero-shot | 1503 | 0.3162 | 0.3150 | -0.0012 [-0.0139, +0.0113] | WITHIN | 0.0535 | -0.0013 | +0.0047 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
