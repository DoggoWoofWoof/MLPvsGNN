# Screen scr-pret (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.6543 | +0.0006 [-0.0024, +0.0038] | WITHIN | 0.0047 | -0.0008 | +0.0012 |
| squad | in-domain | 11873 | 0.9100 | 0.9102 | +0.0003 [-0.0022, +0.0026] | WITHIN | 0.9054 | +0.0003 | -0.0101 |
| musique | zero-shot | 2417 | 0.2696 | 0.2915 | +0.0219 [+0.0100, +0.0346] | GAIN | 0.4734 | +0.0223 | -0.0596 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.8987 | -0.0028 [-0.0052, -0.0004] | WITHIN | 0.6846 | -0.0035 | +0.0108 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8767 | +0.0035 [+0.0016, +0.0053] | WITHIN | 0.6079 | +0.0085 | -0.0034 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.1688 | -0.0090 [-0.0205, +0.0025] | WITHIN | 0.0535 | -0.0073 | -0.0040 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
