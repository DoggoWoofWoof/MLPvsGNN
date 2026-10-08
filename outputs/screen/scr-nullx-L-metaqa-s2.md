# Screen scr-nullx-L-metaqa-s2 (p@swa, s1eval carves)

## Against outputs\step1\fits\L-metaqa (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.0773 | 0.0679 | -0.0095 [-0.0123, -0.0065] | LOSS | 0.0047 | -0.0092 | -0.0019 |
| squad | in-domain | 11873 | 0.9076 | 0.9093 | +0.0017 [-0.0005, +0.0036] | WITHIN | 0.9054 | +0.0017 | +0.0035 |
| musique | in-domain | 2417 | 0.5614 | 0.5604 | -0.0010 [-0.0070, +0.0048] | WITHIN | 0.4734 | -0.0103 | +0.0046 |
| hotpotqa | in-domain | 7405 | 0.9046 | 0.9035 | -0.0011 [-0.0035, +0.0013] | WITHIN | 0.6846 | -0.0028 | +0.0051 |
| 2wiki | in-domain | 12576 | 0.8639 | 0.8696 | +0.0056 [+0.0039, +0.0074] | WITHIN | 0.6079 | +0.0064 | +0.0107 |
| webqsp | zero-shot | 1503 | 0.0926 | 0.0893 | -0.0033 [-0.0112, +0.0039] | WITHIN | 0.0535 | -0.0020 | +0.0000 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
