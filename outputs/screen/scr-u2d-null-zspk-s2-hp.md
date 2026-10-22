# Screen scr-u2d-base-zspk-s2-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zspk-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8043 | 0.8014 | -0.0029 [-0.0039, -0.0020] | WITHIN | 0.0047 | -0.0026 | -0.0090 |
| squad | in-domain | 11873 | 0.9117 | 0.9116 | -0.0002 [-0.0024, +0.0020] | WITHIN | 0.9054 | -0.0002 | +0.0040 |
| musique | in-domain | 2417 | 0.5800 | 0.5845 | +0.0046 [-0.0002, +0.0093] | WITHIN | 0.4734 | +0.0087 | +0.0000 |
| hotpotqa | zero-shot | 7405 | 0.8512 | 0.8614 | +0.0101 [+0.0072, +0.0131] | GAIN | 0.6846 | +0.0188 | +0.0031 |
| 2wiki | in-domain | 12576 | 0.9290 | 0.9261 | -0.0029 [-0.0047, -0.0012] | WITHIN | 0.6079 | -0.0104 | +0.0061 |
| webqsp | zero-shot | 1503 | 0.2773 | 0.2852 | +0.0079 [-0.0010, +0.0164] | WITHIN | 0.0535 | +0.0080 | +0.0007 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
