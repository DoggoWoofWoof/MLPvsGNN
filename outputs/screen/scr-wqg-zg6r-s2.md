# Screen scr-wqg-zg6r-s2 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g6rw-zg6r (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4105 | 0.4085 | -0.0020 [-0.0076, +0.0035] | WITHIN | 0.0047 | +0.0027 | +0.0211 |
| squad | in-domain | 11873 | 0.9105 | 0.9127 | +0.0023 [+0.0003, +0.0042] | WITHIN | 0.9054 | +0.0023 | +0.0027 |
| musique | zero-shot | 2417 | 0.4106 | 0.4618 | +0.0512 [+0.0439, +0.0586] | GAIN | 0.4734 | +0.0323 | +0.0567 |
| hotpotqa | in-domain | 7405 | 0.9134 | 0.9117 | -0.0016 [-0.0034, +0.0002] | WITHIN | 0.6846 | -0.0020 | -0.0038 |
| 2wiki | in-domain | 12576 | 0.9369 | 0.9383 | +0.0014 [+0.0001, +0.0027] | WITHIN | 0.6079 | +0.0033 | +0.0045 |
| webqsp | in-domain | 1503 | 0.6501 | 0.6474 | -0.0026 [-0.0120, +0.0066] | WITHIN | 0.0535 | -0.0027 | +0.0073 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
