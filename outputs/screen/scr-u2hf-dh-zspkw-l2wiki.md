# Screen scr-u2hf-zspkw-l2wiki (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zspk-l2wiki (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8030 | 0.8043 | +0.0013 [+0.0004, +0.0021] | WITHIN | 0.0047 | +0.0025 | +0.0082 |
| squad | in-domain | 11873 | 0.9112 | 0.9113 | +0.0001 [-0.0018, +0.0020] | WITHIN | 0.9054 | +0.0001 | +0.0020 |
| musique | in-domain | 2417 | 0.5708 | 0.5711 | +0.0003 [-0.0049, +0.0060] | WITHIN | 0.4734 | +0.0046 | +0.0070 |
| hotpotqa | in-domain | 7405 | 0.9032 | 0.9057 | +0.0025 [+0.0001, +0.0049] | WITHIN | 0.6846 | +0.0046 | +0.0043 |
| 2wiki | zero-shot | 12576 | 0.8600 | 0.8602 | +0.0002 [-0.0019, +0.0022] | WITHIN | 0.6079 | +0.0003 | +0.0001 |
| webqsp | zero-shot | 1503 | 0.2756 | 0.2886 | +0.0130 [+0.0050, +0.0215] | GAIN | 0.0535 | +0.0113 | +0.0033 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
