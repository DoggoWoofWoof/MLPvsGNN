# Screen scr-a8ex-zrckprqe-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-r1-zrckpr-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8040 | 0.8039 | -0.0002 [-0.0010, +0.0006] | WITHIN | 0.0047 | +0.0000 | +0.0007 |
| squad | in-domain | 11873 | 0.9114 | 0.9109 | -0.0005 [-0.0025, +0.0015] | WITHIN | 0.9054 | -0.0005 | -0.0019 |
| musique | in-domain | 2417 | 0.5779 | 0.5713 | -0.0067 [-0.0117, -0.0010] | WITHIN | 0.4734 | -0.0074 | -0.0029 |
| hotpotqa | zero-shot | 7405 | 0.8558 | 0.8550 | -0.0008 [-0.0039, +0.0020] | WITHIN | 0.6846 | -0.0027 | -0.0063 |
| 2wiki | in-domain | 12576 | 0.9331 | 0.9305 | -0.0026 [-0.0043, -0.0010] | WITHIN | 0.6079 | -0.0050 | +0.0002 |
| webqsp | zero-shot | 1503 | 0.3411 | 0.3726 | +0.0315 [+0.0211, +0.0430] | GAIN | 0.0535 | +0.0273 | +0.0160 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8039 | +0.1499 [+0.1442, +0.1552] | GAIN | 0.0047 | +0.1534 | +0.2861 |
| squad | in-domain | 11873 | 0.9116 | 0.9109 | -0.0008 [-0.0033, +0.0016] | WITHIN | 0.9054 | -0.0008 | +0.0008 |
| musique | in-domain | 2417 | 0.5550 | 0.5713 | +0.0162 [+0.0083, +0.0240] | GAIN | 0.4734 | +0.0145 | +0.0310 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8550 | -0.0035 [-0.0084, +0.0009] | WITHIN | 0.6846 | -0.0082 | -0.0315 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9305 | +0.0633 [+0.0600, +0.0669] | GAIN | 0.6079 | +0.1300 | +0.0627 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3726 | +0.2526 [+0.2272, +0.2776] | GAIN | 0.0535 | +0.2069 | +0.0805 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
