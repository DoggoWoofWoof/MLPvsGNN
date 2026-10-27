# Screen scr-a4fx-zrckprf (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-r1-zrckpr (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8012 | 0.7994 | -0.0018 [-0.0032, -0.0004] | WITHIN | 0.0047 | -0.0017 | -0.0250 |
| squad | in-domain | 11873 | 0.9127 | 0.9097 | -0.0030 [-0.0051, -0.0010] | WITHIN | 0.9054 | -0.0030 | -0.0039 |
| musique | zero-shot | 2417 | 0.5197 | 0.3534 | -0.1663 [-0.1766, -0.1560] | LOSS | 0.4734 | -0.0956 | -0.3240 |
| hotpotqa | in-domain | 7405 | 0.9083 | 0.9038 | -0.0045 [-0.0069, -0.0022] | WITHIN | 0.6846 | -0.0077 | -0.0070 |
| 2wiki | in-domain | 12576 | 0.9361 | 0.9310 | -0.0051 [-0.0067, -0.0035] | WITHIN | 0.6079 | -0.0149 | -0.0041 |
| webqsp | zero-shot | 1503 | 0.3762 | 0.3542 | -0.0220 [-0.0345, -0.0100] | LOSS | 0.0535 | -0.0220 | -0.0226 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7994 | +0.1457 [+0.1402, +0.1507] | GAIN | 0.0047 | +0.1434 | +0.2580 |
| squad | in-domain | 11873 | 0.9100 | 0.9097 | -0.0003 [-0.0026, +0.0020] | WITHIN | 0.9054 | -0.0003 | -0.0068 |
| musique | zero-shot | 2417 | 0.2696 | 0.3534 | +0.0839 [+0.0734, +0.0948] | GAIN | 0.4734 | +0.0348 | +0.0898 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9038 | +0.0023 [-0.0010, +0.0053] | WITHIN | 0.6846 | +0.0016 | -0.0146 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9310 | +0.0577 [+0.0548, +0.0609] | GAIN | 0.6079 | +0.1159 | +0.0663 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3542 | +0.1763 [+0.1564, +0.1991] | GAIN | 0.0535 | +0.1437 | +0.0605 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
