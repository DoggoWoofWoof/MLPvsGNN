# Screen scr-u2d-new-zspkv-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-new-zspk-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7999 | 0.7990 | -0.0009 [-0.0019, +0.0001] | WITHIN | 0.0047 | -0.0006 | -0.0007 |
| squad | in-domain | 11873 | 0.9159 | 0.9143 | -0.0016 [-0.0033, -0.0001] | WITHIN | 0.9054 | -0.0016 | +0.0021 |
| musique | in-domain | 2417 | 0.5765 | 0.5727 | -0.0038 [-0.0086, +0.0013] | WITHIN | 0.4734 | -0.0066 | -0.0165 |
| hotpotqa | zero-shot | 7405 | 0.8018 | 0.8086 | +0.0068 [+0.0039, +0.0096] | WITHIN | 0.6846 | +0.0132 | +0.0031 |
| 2wiki | in-domain | 12576 | 0.9095 | 0.9107 | +0.0012 [-0.0006, +0.0030] | WITHIN | 0.6079 | +0.0058 | -0.0010 |
| webqsp | zero-shot | 1503 | 0.2942 | 0.2867 | -0.0075 [-0.0165, +0.0015] | WITHIN | 0.0535 | -0.0067 | +0.0033 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
