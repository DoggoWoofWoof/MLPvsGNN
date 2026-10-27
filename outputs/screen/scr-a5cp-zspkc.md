# Screen scr-a5cp-zspkc (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1gb-zspk (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8004 | 0.7981 | -0.0023 [-0.0035, -0.0011] | WITHIN | 0.0047 | -0.0009 | -0.0048 |
| squad | in-domain | 11873 | 0.9129 | 0.9132 | +0.0003 [-0.0013, +0.0019] | WITHIN | 0.9054 | +0.0003 | -0.0019 |
| musique | zero-shot | 2417 | 0.5169 | 0.5300 | +0.0132 [+0.0079, +0.0183] | GAIN | 0.4734 | +0.0157 | +0.0017 |
| hotpotqa | in-domain | 7405 | 0.9086 | 0.9090 | +0.0004 [-0.0018, +0.0025] | WITHIN | 0.6846 | +0.0018 | +0.0115 |
| 2wiki | in-domain | 12576 | 0.9344 | 0.9365 | +0.0021 [+0.0008, +0.0036] | WITHIN | 0.6079 | +0.0069 | +0.0002 |
| webqsp | zero-shot | 1503 | 0.3354 | 0.3473 | +0.0119 [+0.0000, +0.0244] | GAIN | 0.0535 | +0.0133 | +0.0080 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7981 | +0.1443 [+0.1389, +0.1494] | GAIN | 0.0047 | +0.1447 | +0.2741 |
| squad | in-domain | 11873 | 0.9100 | 0.9132 | +0.0033 [+0.0010, +0.0056] | WITHIN | 0.9054 | +0.0033 | -0.0015 |
| musique | zero-shot | 2417 | 0.2696 | 0.5300 | +0.2604 [+0.2487, +0.2727] | GAIN | 0.4734 | +0.1481 | +0.4191 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9090 | +0.0076 [+0.0043, +0.0107] | GAIN | 0.6846 | +0.0113 | -0.0043 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9365 | +0.0633 [+0.0601, +0.0666] | GAIN | 0.6079 | +0.1306 | +0.0712 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3473 | +0.1694 [+0.1451, +0.1933] | GAIN | 0.0535 | +0.1344 | +0.1317 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
