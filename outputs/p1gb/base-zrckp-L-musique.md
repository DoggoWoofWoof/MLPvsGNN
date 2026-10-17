# Screen scr-p1c3p-zrckp (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8019 | +0.1482 [+0.1426, +0.1535] | GAIN | 0.0047 | +0.1454 | +0.2811 |
| squad | in-domain | 11873 | 0.9100 | 0.9116 | +0.0017 [-0.0006, +0.0039] | WITHIN | 0.9054 | +0.0017 | -0.0011 |
| musique | zero-shot | 2417 | 0.2696 | 0.5121 | +0.2426 [+0.2305, +0.2546] | GAIN | 0.4734 | +0.1208 | +0.4133 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9065 | +0.0051 [+0.0017, +0.0084] | WITHIN | 0.6846 | +0.0042 | -0.0074 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9347 | +0.0615 [+0.0582, +0.0647] | GAIN | 0.6079 | +0.1263 | +0.0727 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3627 | +0.1849 [+0.1639, +0.2079] | GAIN | 0.0535 | +0.1583 | +0.0725 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
