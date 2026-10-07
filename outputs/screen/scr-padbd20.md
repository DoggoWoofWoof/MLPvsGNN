# Screen scr-padbd20 (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.6484 | -0.0053 [-0.0085, -0.0022] | WITHIN | 0.0047 | -0.0060 | -0.0088 |
| squad | in-domain | 11873 | 0.9100 | 0.9110 | +0.0010 [-0.0012, +0.0033] | WITHIN | 0.9054 | +0.0010 | -0.0036 |
| musique | zero-shot | 2417 | 0.2696 | 0.3791 | +0.1095 [+0.0991, +0.1193] | GAIN | 0.4734 | +0.0583 | +0.1221 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.8959 | -0.0055 [-0.0080, -0.0031] | WITHIN | 0.6846 | -0.0120 | -0.0012 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8664 | -0.0069 [-0.0089, -0.0048] | WITHIN | 0.6079 | -0.0119 | -0.0082 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.1667 | -0.0111 [-0.0224, +0.0001] | WITHIN | 0.0535 | -0.0053 | -0.0146 |

## Against outputs\screen\fits\scr-bdrop20 (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6491 | 0.6484 | -0.0007 [-0.0035, +0.0022] | WITHIN | 0.0047 | +0.0010 | -0.0061 |
| squad | in-domain | 11873 | 0.9093 | 0.9110 | +0.0017 [-0.0003, +0.0037] | WITHIN | 0.9054 | +0.0017 | -0.0014 |
| musique | zero-shot | 2417 | 0.4039 | 0.3791 | -0.0248 [-0.0337, -0.0157] | LOSS | 0.4734 | -0.0083 | -0.0492 |
| hotpotqa | in-domain | 7405 | 0.8967 | 0.8959 | -0.0007 [-0.0030, +0.0017] | WITHIN | 0.6846 | -0.0019 | +0.0059 |
| 2wiki | in-domain | 12576 | 0.8682 | 0.8664 | -0.0018 [-0.0039, +0.0001] | WITHIN | 0.6079 | -0.0010 | +0.0039 |
| webqsp | zero-shot | 1503 | 0.1742 | 0.1667 | -0.0074 [-0.0191, +0.0046] | WITHIN | 0.0535 | -0.0040 | -0.0106 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
