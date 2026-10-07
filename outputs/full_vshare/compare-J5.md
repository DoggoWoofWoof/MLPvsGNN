# Screen J5 (p@swa, s1eval carves)

## Against outputs\step1\fits\J5 (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6443 | 0.6466 | +0.0023 [-0.0006, +0.0051] | WITHIN | 0.0047 | +0.0049 | -0.0030 |
| squad | in-domain | 11873 | 0.9095 | 0.9064 | -0.0031 [-0.0051, -0.0012] | WITHIN | 0.9054 | -0.0031 | -0.0042 |
| musique | in-domain | 2417 | 0.5609 | 0.5501 | -0.0108 [-0.0174, -0.0042] | LOSS | 0.4734 | -0.0112 | -0.0149 |
| hotpotqa | in-domain | 7405 | 0.9013 | 0.8982 | -0.0030 [-0.0056, -0.0004] | WITHIN | 0.6846 | -0.0051 | +0.0000 |
| 2wiki | in-domain | 12576 | 0.8694 | 0.8684 | -0.0010 [-0.0030, +0.0011] | WITHIN | 0.6079 | -0.0006 | -0.0016 |
| webqsp | zero-shot | 1503 | 0.1167 | 0.1098 | -0.0070 [-0.0162, +0.0016] | WITHIN | 0.0535 | -0.0060 | +0.0000 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
