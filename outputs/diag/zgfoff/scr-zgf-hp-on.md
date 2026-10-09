# Screen scr-zgf-hp-on (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zgf-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7815 | 0.7815 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.0047 | +0.0000 | +0.0000 |
| squad | in-domain | 11873 | 0.8972 | 0.8972 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.9054 | +0.0000 | +0.0000 |
| musique | in-domain | 2417 | 0.5001 | 0.5001 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.4734 | +0.0000 | +0.0000 |
| hotpotqa | zero-shot | 7405 | 0.8020 | 0.8020 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.6846 | +0.0000 | +0.0000 |
| 2wiki | in-domain | 12576 | 0.8626 | 0.8626 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.6079 | +0.0000 | +0.0000 |
| webqsp | zero-shot | 1503 | 0.2407 | 0.2407 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.0535 | +0.0000 | +0.0000 |

## Against outputs\screen\fits\scr-zrm-hp (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7801 | 0.7815 | +0.0015 [+0.0002, +0.0028] | WITHIN | 0.0047 | -0.0014 | +0.0081 |
| squad | in-domain | 11873 | 0.9116 | 0.8972 | -0.0143 [-0.0175, -0.0112] | LOSS | 0.9054 | -0.0143 | -0.0286 |
| musique | in-domain | 2417 | 0.5600 | 0.5001 | -0.0599 [-0.0683, -0.0511] | LOSS | 0.4734 | -0.0662 | -0.1361 |
| hotpotqa | zero-shot | 7405 | 0.8457 | 0.8020 | -0.0437 [-0.0481, -0.0392] | LOSS | 0.6846 | -0.0810 | -0.0570 |
| 2wiki | in-domain | 12576 | 0.8681 | 0.8626 | -0.0055 [-0.0081, -0.0028] | WITHIN | 0.6079 | -0.0095 | -0.0358 |
| webqsp | zero-shot | 1503 | 0.2416 | 0.2407 | -0.0008 [-0.0122, +0.0108] | WITHIN | 0.0535 | -0.0033 | +0.0040 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
