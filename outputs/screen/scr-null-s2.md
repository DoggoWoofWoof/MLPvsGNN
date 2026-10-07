# Screen scr-null-s2 (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.6514 | -0.0023 [-0.0052, +0.0006] | WITHIN | 0.0047 | -0.0035 | -0.0008 |
| squad | in-domain | 11873 | 0.9100 | 0.9112 | +0.0013 [-0.0008, +0.0034] | WITHIN | 0.9054 | +0.0013 | -0.0008 |
| musique | zero-shot | 2417 | 0.2696 | 0.2617 | -0.0079 [-0.0151, -0.0002] | LOSS | 0.4734 | -0.0037 | -0.0178 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9031 | +0.0016 [-0.0006, +0.0040] | WITHIN | 0.6846 | +0.0034 | +0.0036 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8726 | -0.0007 [-0.0025, +0.0011] | WITHIN | 0.6079 | -0.0012 | -0.0002 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.1761 | -0.0018 [-0.0107, +0.0076] | WITHIN | 0.0535 | +0.0020 | -0.0060 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
