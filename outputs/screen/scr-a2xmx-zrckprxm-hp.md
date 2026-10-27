# Screen scr-a2xmx-zrckprxm-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-r1-zrckpr-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8040 | 0.7977 | -0.0064 [-0.0076, -0.0052] | WITHIN | 0.0047 | -0.0050 | -0.0164 |
| squad | in-domain | 11873 | 0.9114 | 0.9114 | +0.0000 [-0.0019, +0.0019] | WITHIN | 0.9054 | +0.0000 | -0.0024 |
| musique | in-domain | 2417 | 0.5779 | 0.5723 | -0.0057 [-0.0120, +0.0000] | WITHIN | 0.4734 | -0.0058 | +0.0054 |
| hotpotqa | zero-shot | 7405 | 0.8558 | 0.8540 | -0.0018 [-0.0051, +0.0015] | WITHIN | 0.6846 | -0.0028 | +0.0031 |
| 2wiki | in-domain | 12576 | 0.9331 | 0.9243 | -0.0088 [-0.0108, -0.0070] | LOSS | 0.6079 | -0.0270 | +0.0068 |
| webqsp | zero-shot | 1503 | 0.3411 | 0.3382 | -0.0029 [-0.0163, +0.0103] | WITHIN | 0.0535 | -0.0007 | +0.0047 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7977 | +0.1437 [+0.1383, +0.1490] | GAIN | 0.0047 | +0.1484 | +0.2690 |
| squad | in-domain | 11873 | 0.9116 | 0.9114 | -0.0003 [-0.0025, +0.0019] | WITHIN | 0.9054 | -0.0003 | +0.0003 |
| musique | in-domain | 2417 | 0.5550 | 0.5723 | +0.0172 [+0.0096, +0.0248] | GAIN | 0.4734 | +0.0161 | +0.0393 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8540 | -0.0045 [-0.0092, +0.0001] | WITHIN | 0.6846 | -0.0084 | -0.0220 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9243 | +0.0571 [+0.0538, +0.0606] | GAIN | 0.6079 | +0.1081 | +0.0693 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3382 | +0.2182 [+0.1937, +0.2417] | GAIN | 0.0535 | +0.1790 | +0.0692 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
