# Screen scr-r1-zrckpr-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3p-zrckp-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8044 | 0.8040 | -0.0004 [-0.0012, +0.0005] | WITHIN | 0.0047 | -0.0005 | -0.0031 |
| squad | in-domain | 11873 | 0.9127 | 0.9114 | -0.0013 [-0.0033, +0.0006] | WITHIN | 0.9054 | -0.0013 | +0.0002 |
| musique | in-domain | 2417 | 0.5745 | 0.5779 | +0.0034 [-0.0015, +0.0086] | WITHIN | 0.4734 | +0.0037 | +0.0008 |
| hotpotqa | zero-shot | 7405 | 0.8513 | 0.8558 | +0.0045 [+0.0016, +0.0074] | WITHIN | 0.6846 | +0.0096 | +0.0093 |
| 2wiki | in-domain | 12576 | 0.9350 | 0.9331 | -0.0019 [-0.0034, -0.0003] | WITHIN | 0.6079 | -0.0060 | -0.0021 |
| webqsp | zero-shot | 1503 | 0.3475 | 0.3411 | -0.0064 [-0.0156, +0.0028] | WITHIN | 0.0535 | -0.0053 | -0.0040 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8040 | +0.1501 [+0.1445, +0.1554] | GAIN | 0.0047 | +0.1534 | +0.2853 |
| squad | in-domain | 11873 | 0.9116 | 0.9114 | -0.0003 [-0.0025, +0.0020] | WITHIN | 0.9054 | -0.0003 | +0.0026 |
| musique | in-domain | 2417 | 0.5550 | 0.5779 | +0.0229 [+0.0148, +0.0305] | GAIN | 0.4734 | +0.0219 | +0.0339 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8558 | -0.0027 [-0.0074, +0.0016] | WITHIN | 0.6846 | -0.0055 | -0.0251 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9331 | +0.0659 [+0.0626, +0.0693] | GAIN | 0.6079 | +0.1350 | +0.0624 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3411 | +0.2211 [+0.1974, +0.2464] | GAIN | 0.0535 | +0.1796 | +0.0645 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
