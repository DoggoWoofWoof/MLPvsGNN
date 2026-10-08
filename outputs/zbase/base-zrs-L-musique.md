# Screen scr-zrs (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7637 | +0.1099 [+0.1052, +0.1147] | GAIN | 0.0047 | +0.1070 | +0.1900 |
| squad | in-domain | 11873 | 0.9100 | 0.9106 | +0.0007 [-0.0014, +0.0028] | WITHIN | 0.9054 | +0.0007 | -0.0040 |
| musique | zero-shot | 2417 | 0.2696 | 0.3897 | +0.1202 [+0.1102, +0.1301] | GAIN | 0.4734 | +0.0683 | +0.1539 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9061 | +0.0046 [+0.0021, +0.0072] | WITHIN | 0.6846 | +0.0076 | +0.0034 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8714 | -0.0018 [-0.0038, +0.0002] | WITHIN | 0.6079 | -0.0055 | +0.0041 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.2623 | +0.0845 [+0.0671, +0.1025] | GAIN | 0.0535 | +0.0659 | +0.0592 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
