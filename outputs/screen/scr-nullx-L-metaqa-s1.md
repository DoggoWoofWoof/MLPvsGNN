# Screen scr-nullx-L-metaqa-s1 (p@swa, s1eval carves)

## Against outputs\step1\fits\L-metaqa (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.0773 | 0.0844 | +0.0071 [+0.0045, +0.0098] | WITHIN | 0.0047 | +0.0049 | -0.0003 |
| squad | in-domain | 11873 | 0.9076 | 0.9106 | +0.0029 [+0.0008, +0.0051] | WITHIN | 0.9054 | +0.0029 | +0.0020 |
| musique | in-domain | 2417 | 0.5614 | 0.5587 | -0.0027 [-0.0092, +0.0034] | WITHIN | 0.4734 | -0.0025 | +0.0132 |
| hotpotqa | in-domain | 7405 | 0.9046 | 0.9039 | -0.0007 [-0.0033, +0.0017] | WITHIN | 0.6846 | -0.0003 | +0.0062 |
| 2wiki | in-domain | 12576 | 0.8639 | 0.8750 | +0.0110 [+0.0091, +0.0129] | GAIN | 0.6079 | +0.0169 | +0.0086 |
| webqsp | zero-shot | 1503 | 0.0926 | 0.0946 | +0.0020 [-0.0063, +0.0098] | WITHIN | 0.0535 | +0.0027 | +0.0013 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
