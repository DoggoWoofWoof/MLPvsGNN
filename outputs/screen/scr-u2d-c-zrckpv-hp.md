# Screen scr-u2d-new-zrckpv-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zrckp-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8033 | 0.8002 | -0.0031 [-0.0048, -0.0016] | WITHIN | 0.0047 | -0.0030 | -0.0053 |
| squad | in-domain | 11873 | 0.9124 | 0.9128 | +0.0004 [-0.0019, +0.0029] | WITHIN | 0.9054 | +0.0004 | -0.0017 |
| musique | in-domain | 2417 | 0.5697 | 0.5737 | +0.0040 [-0.0024, +0.0108] | WITHIN | 0.4734 | +0.0008 | +0.0066 |
| hotpotqa | zero-shot | 7405 | 0.8538 | 0.8087 | -0.0451 [-0.0499, -0.0406] | LOSS | 0.6846 | -0.0883 | +0.0215 |
| 2wiki | in-domain | 12576 | 0.9303 | 0.9060 | -0.0243 [-0.0272, -0.0212] | LOSS | 0.6079 | -0.0587 | +0.0058 |
| webqsp | zero-shot | 1503 | 0.3370 | 0.3520 | +0.0150 [+0.0007, +0.0305] | GAIN | 0.0535 | +0.0073 | +0.0027 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
