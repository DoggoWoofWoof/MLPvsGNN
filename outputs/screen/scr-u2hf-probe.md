# Screen scr-u2hf-probe-l4 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-probe-t4 (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8018 | 0.8005 | -0.0013 [-0.0023, -0.0002] | WITHIN | 0.0047 | -0.0008 | -0.0015 |
| squad | in-domain | 11873 | 0.9116 | 0.9118 | +0.0002 [-0.0015, +0.0019] | WITHIN | 0.9054 | +0.0002 | +0.0011 |
| musique | in-domain | 2417 | 0.5737 | 0.5783 | +0.0046 [-0.0005, +0.0093] | WITHIN | 0.4734 | +0.0021 | +0.0029 |
| hotpotqa | in-domain | 7405 | 0.9066 | 0.9074 | +0.0008 [-0.0011, +0.0028] | WITHIN | 0.6846 | -0.0005 | +0.0131 |
| 2wiki | in-domain | 12576 | 0.9357 | 0.9352 | -0.0006 [-0.0020, +0.0010] | WITHIN | 0.6079 | -0.0014 | +0.0011 |
| webqsp | zero-shot | 1503 | 0.3643 | 0.3503 | -0.0140 [-0.0236, -0.0047] | LOSS | 0.0535 | -0.0113 | -0.0053 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
