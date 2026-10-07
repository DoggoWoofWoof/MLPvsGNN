# Screen scr-bdrop20 (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.6491 | -0.0047 [-0.0075, -0.0018] | WITHIN | 0.0047 | -0.0071 | -0.0027 |
| squad | in-domain | 11873 | 0.9100 | 0.9093 | -0.0007 [-0.0028, +0.0015] | WITHIN | 0.9054 | -0.0007 | -0.0022 |
| musique | zero-shot | 2417 | 0.2696 | 0.4039 | +0.1344 [+0.1236, +0.1447] | GAIN | 0.4734 | +0.0666 | +0.1713 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.8967 | -0.0048 [-0.0073, -0.0025] | WITHIN | 0.6846 | -0.0101 | -0.0072 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8682 | -0.0050 [-0.0070, -0.0031] | WITHIN | 0.6079 | -0.0110 | -0.0121 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.1742 | -0.0037 [-0.0159, +0.0088] | WITHIN | 0.0535 | -0.0013 | -0.0040 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
