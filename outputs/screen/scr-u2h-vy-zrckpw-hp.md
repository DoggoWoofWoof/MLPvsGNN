# Screen scr-u2h-zrckpw-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2g-y-zrckp-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8046 | 0.8043 | -0.0003 [-0.0012, +0.0006] | WITHIN | 0.0047 | -0.0009 | -0.0006 |
| squad | in-domain | 11873 | 0.9146 | 0.9105 | -0.0041 [-0.0061, -0.0021] | WITHIN | 0.9054 | -0.0041 | -0.0036 |
| musique | in-domain | 2417 | 0.5766 | 0.5736 | -0.0030 [-0.0091, +0.0026] | WITHIN | 0.4734 | +0.0046 | -0.0170 |
| hotpotqa | zero-shot | 7405 | 0.8055 | 0.8469 | +0.0415 [+0.0373, +0.0459] | GAIN | 0.6846 | +0.0817 | -0.0039 |
| 2wiki | in-domain | 12576 | 0.9218 | 0.9381 | +0.0163 [+0.0138, +0.0191] | GAIN | 0.6079 | +0.0413 | -0.0116 |
| webqsp | zero-shot | 1503 | 0.3369 | 0.3312 | -0.0057 [-0.0172, +0.0064] | WITHIN | 0.0535 | -0.0013 | -0.0153 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
