# Screen scr-u2d-new-zspkv (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-new-zspk (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7991 | 0.7997 | +0.0006 [-0.0005, +0.0016] | WITHIN | 0.0047 | +0.0002 | -0.0040 |
| squad | in-domain | 11873 | 0.9137 | 0.9127 | -0.0009 [-0.0028, +0.0008] | WITHIN | 0.9054 | -0.0009 | +0.0002 |
| musique | zero-shot | 2417 | 0.5257 | 0.5264 | +0.0007 [-0.0043, +0.0053] | WITHIN | 0.4734 | -0.0029 | -0.0215 |
| hotpotqa | in-domain | 7405 | 0.8483 | 0.8464 | -0.0020 [-0.0045, +0.0005] | WITHIN | 0.6846 | -0.0043 | -0.0001 |
| 2wiki | in-domain | 12576 | 0.9187 | 0.9159 | -0.0028 [-0.0043, -0.0012] | WITHIN | 0.6079 | -0.0107 | -0.0047 |
| webqsp | zero-shot | 1503 | 0.3223 | 0.3150 | -0.0072 [-0.0153, +0.0010] | WITHIN | 0.0535 | -0.0033 | -0.0093 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
