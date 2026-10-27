# Screen scr-a4fx-zrckprf-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-r1-zrckpr-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8040 | 0.7968 | -0.0073 [-0.0088, -0.0059] | WITHIN | 0.0047 | -0.0097 | -0.0273 |
| squad | in-domain | 11873 | 0.9114 | 0.9109 | -0.0005 [-0.0029, +0.0018] | WITHIN | 0.9054 | -0.0005 | -0.0006 |
| musique | in-domain | 2417 | 0.5779 | 0.5702 | -0.0077 [-0.0134, -0.0020] | LOSS | 0.4734 | -0.0108 | -0.0062 |
| hotpotqa | zero-shot | 7405 | 0.8558 | 0.8612 | +0.0054 [+0.0022, +0.0086] | WITHIN | 0.6846 | +0.0107 | +0.0014 |
| 2wiki | in-domain | 12576 | 0.9331 | 0.9281 | -0.0050 [-0.0068, -0.0032] | WITHIN | 0.6079 | -0.0111 | -0.0068 |
| webqsp | zero-shot | 1503 | 0.3411 | 0.3531 | +0.0120 [-0.0015, +0.0250] | WITHIN | 0.0535 | +0.0053 | -0.0060 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7968 | +0.1428 [+0.1376, +0.1479] | GAIN | 0.0047 | +0.1437 | +0.2580 |
| squad | in-domain | 11873 | 0.9116 | 0.9109 | -0.0008 [-0.0034, +0.0017] | WITHIN | 0.9054 | -0.0008 | +0.0020 |
| musique | in-domain | 2417 | 0.5550 | 0.5702 | +0.0152 [+0.0080, +0.0228] | GAIN | 0.4734 | +0.0112 | +0.0277 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8612 | +0.0027 [-0.0021, +0.0072] | WITHIN | 0.6846 | +0.0051 | -0.0238 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9281 | +0.0609 [+0.0576, +0.0644] | GAIN | 0.6079 | +0.1239 | +0.0556 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3531 | +0.2331 [+0.2079, +0.2572] | GAIN | 0.0535 | +0.1850 | +0.0585 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
