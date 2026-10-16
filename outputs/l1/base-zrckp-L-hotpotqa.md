# Screen scr-p1c3p-zrckp-hp (p@swa, s1eval carves)

## Against outputs\step1\fits\L-hotpotqa (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8044 | +0.1504 [+0.1447, +0.1558] | GAIN | 0.0047 | +0.1539 | +0.2884 |
| squad | in-domain | 11873 | 0.9116 | 0.9127 | +0.0011 [-0.0010, +0.0033] | WITHIN | 0.9054 | +0.0011 | +0.0024 |
| musique | in-domain | 2417 | 0.5550 | 0.5745 | +0.0194 [+0.0122, +0.0269] | GAIN | 0.4734 | +0.0182 | +0.0331 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8513 | -0.0072 [-0.0120, -0.0026] | WITHIN | 0.6846 | -0.0151 | -0.0344 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9350 | +0.0678 [+0.0645, +0.0712] | GAIN | 0.6079 | +0.1410 | +0.0646 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3475 | +0.2275 [+0.2033, +0.2523] | GAIN | 0.0535 | +0.1850 | +0.0685 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
