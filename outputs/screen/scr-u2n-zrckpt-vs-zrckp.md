# Screen scr-u2n-zrckpt (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zrckp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7995 | 0.7993 | -0.0002 [-0.0011, +0.0006] | WITHIN | 0.0047 | -0.0005 | +0.0036 |
| squad | in-domain | 11873 | 0.9116 | 0.9116 | +0.0000 [-0.0018, +0.0017] | WITHIN | 0.9054 | +0.0000 | +0.0010 |
| musique | zero-shot | 2417 | 0.5129 | 0.5228 | +0.0098 [+0.0055, +0.0144] | GAIN | 0.4734 | +0.0108 | +0.0182 |
| hotpotqa | in-domain | 7405 | 0.9041 | 0.9066 | +0.0026 [+0.0005, +0.0046] | WITHIN | 0.6846 | +0.0054 | -0.0076 |
| 2wiki | in-domain | 12576 | 0.9319 | 0.9361 | +0.0042 [+0.0026, +0.0056] | WITHIN | 0.6079 | +0.0103 | -0.0002 |
| webqsp | zero-shot | 1503 | 0.3570 | 0.3517 | -0.0053 [-0.0137, +0.0034] | WITHIN | 0.0535 | -0.0047 | +0.0140 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
