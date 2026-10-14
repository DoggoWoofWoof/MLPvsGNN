# Screen scr-p1c2-zsp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zspg (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7794 | 0.8008 | +0.0214 [+0.0185, +0.0243] | GAIN | 0.0047 | +0.0188 | +0.0293 |
| squad | in-domain | 11873 | 0.9125 | 0.9119 | -0.0006 [-0.0027, +0.0015] | WITHIN | 0.9054 | -0.0006 | +0.0008 |
| musique | zero-shot | 2417 | 0.5111 | 0.4792 | -0.0320 [-0.0392, -0.0242] | LOSS | 0.4734 | -0.0463 | -0.1341 |
| hotpotqa | in-domain | 7405 | 0.9065 | 0.9062 | -0.0003 [-0.0032, +0.0026] | WITHIN | 0.6846 | -0.0003 | +0.0058 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.9174 | +0.0450 [+0.0422, +0.0478] | GAIN | 0.6079 | +0.0861 | +0.0400 |
| webqsp | zero-shot | 1503 | 0.3245 | 0.3078 | -0.0167 [-0.0333, -0.0003] | LOSS | 0.0535 | -0.0160 | -0.0133 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8008 | +0.1471 [+0.1414, +0.1526] | GAIN | 0.0047 | +0.1463 | +0.2829 |
| squad | in-domain | 11873 | 0.9100 | 0.9119 | +0.0019 [-0.0003, +0.0041] | WITHIN | 0.9054 | +0.0019 | -0.0024 |
| musique | zero-shot | 2417 | 0.2696 | 0.4792 | +0.2096 [+0.1985, +0.2208] | GAIN | 0.4734 | +0.0906 | +0.2213 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9062 | +0.0047 [+0.0018, +0.0078] | WITHIN | 0.6846 | +0.0078 | +0.0189 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9174 | +0.0442 [+0.0414, +0.0470] | GAIN | 0.6079 | +0.0883 | +0.0378 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3078 | +0.1300 [+0.1077, +0.1537] | GAIN | 0.0535 | +0.1005 | +0.1011 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
