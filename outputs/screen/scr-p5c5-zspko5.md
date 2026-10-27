# Screen scr-p5c5-zspko5 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3-zspk (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8009 | 0.8052 | +0.0043 [+0.0033, +0.0054] | WITHIN | 0.0047 | +0.0039 | +0.0196 |
| squad | in-domain | 11873 | 0.9114 | 0.9106 | -0.0008 [-0.0029, +0.0012] | WITHIN | 0.9054 | -0.0008 | -0.0012 |
| musique | zero-shot | 2417 | 0.5012 | 0.5158 | +0.0146 [+0.0087, +0.0208] | GAIN | 0.4734 | +0.0116 | +0.0037 |
| hotpotqa | in-domain | 7405 | 0.9080 | 0.9088 | +0.0007 [-0.0014, +0.0028] | WITHIN | 0.6846 | -0.0003 | +0.0012 |
| 2wiki | in-domain | 12576 | 0.9383 | 0.9369 | -0.0014 [-0.0027, -0.0000] | WITHIN | 0.6079 | -0.0024 | -0.0053 |
| webqsp | zero-shot | 1503 | 0.3229 | 0.3086 | -0.0142 [-0.0250, -0.0034] | LOSS | 0.0535 | -0.0126 | +0.0040 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8052 | +0.1515 [+0.1459, +0.1569] | GAIN | 0.0047 | +0.1492 | +0.2945 |
| squad | in-domain | 11873 | 0.9100 | 0.9106 | +0.0006 [-0.0016, +0.0029] | WITHIN | 0.9054 | +0.0006 | -0.0027 |
| musique | zero-shot | 2417 | 0.2696 | 0.5158 | +0.2462 [+0.2346, +0.2579] | GAIN | 0.4734 | +0.1278 | +0.3906 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9088 | +0.0073 [+0.0040, +0.0107] | WITHIN | 0.6846 | +0.0096 | +0.0062 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9369 | +0.0637 [+0.0605, +0.0668] | GAIN | 0.6079 | +0.1330 | +0.0670 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3086 | +0.1308 [+0.1083, +0.1555] | GAIN | 0.0535 | +0.0978 | +0.1317 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
