# Screen J5 (p@swa, s1eval carves)

## Against outputs\step1\fits\J5 (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6443 | 0.7788 | +0.1345 [+0.1292, +0.1400] | GAIN | 0.0047 | +0.1409 | +0.2540 |
| squad | in-domain | 11873 | 0.9095 | 0.9075 | -0.0020 [-0.0040, -0.0001] | WITHIN | 0.9054 | -0.0020 | -0.0030 |
| musique | in-domain | 2417 | 0.5609 | 0.5685 | +0.0077 [+0.0024, +0.0132] | GAIN | 0.4734 | +0.0161 | +0.0062 |
| hotpotqa | in-domain | 7405 | 0.9013 | 0.9026 | +0.0014 [-0.0011, +0.0038] | WITHIN | 0.6846 | +0.0030 | +0.0043 |
| 2wiki | in-domain | 12576 | 0.8694 | 0.8730 | +0.0036 [+0.0018, +0.0055] | WITHIN | 0.6079 | +0.0052 | -0.0017 |
| webqsp | zero-shot | 1503 | 0.1167 | 0.2795 | +0.1628 [+0.1393, +0.1858] | GAIN | 0.0535 | +0.1277 | +0.0892 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
