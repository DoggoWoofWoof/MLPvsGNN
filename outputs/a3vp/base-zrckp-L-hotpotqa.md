# Screen scr-p1gb-zrckp-hp (p@swa, s1eval carves)

## Against outputs\step1\fits\L-hotpotqa (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7997 | +0.1457 [+0.1402, +0.1509] | GAIN | 0.0047 | +0.1498 | +0.2774 |
| squad | in-domain | 11873 | 0.9116 | 0.9122 | +0.0006 [-0.0018, +0.0030] | WITHIN | 0.9054 | +0.0006 | +0.0010 |
| musique | in-domain | 2417 | 0.5550 | 0.5695 | +0.0145 [+0.0061, +0.0228] | GAIN | 0.4734 | +0.0190 | +0.0319 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8546 | -0.0039 [-0.0084, +0.0003] | WITHIN | 0.6846 | -0.0101 | -0.0379 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9340 | +0.0668 [+0.0637, +0.0702] | GAIN | 0.6079 | +0.1352 | +0.0638 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3364 | +0.2164 [+0.1918, +0.2404] | GAIN | 0.0535 | +0.1730 | +0.0659 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
