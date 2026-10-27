# Screen scr-p1gb-zrckp (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7993 | +0.1455 [+0.1400, +0.1507] | GAIN | 0.0047 | +0.1447 | +0.2684 |
| squad | in-domain | 11873 | 0.9100 | 0.9127 | +0.0027 [+0.0007, +0.0049] | WITHIN | 0.9054 | +0.0027 | -0.0022 |
| musique | zero-shot | 2417 | 0.2696 | 0.5013 | +0.2317 [+0.2200, +0.2436] | GAIN | 0.4734 | +0.1163 | +0.4199 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9035 | +0.0020 [-0.0015, +0.0055] | WITHIN | 0.6846 | +0.0014 | -0.0248 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9377 | +0.0645 [+0.0612, +0.0677] | GAIN | 0.6079 | +0.1346 | +0.0710 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3768 | +0.1990 [+0.1769, +0.2222] | GAIN | 0.0535 | +0.1710 | +0.0945 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
