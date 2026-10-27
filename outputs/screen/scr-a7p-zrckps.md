# Screen scr-a7p-zrckps (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1gb-zrckp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7993 | 0.8014 | +0.0022 [+0.0011, +0.0033] | WITHIN | 0.0047 | +0.0018 | +0.0115 |
| squad | in-domain | 11873 | 0.9127 | 0.9126 | -0.0001 [-0.0019, +0.0016] | WITHIN | 0.9054 | -0.0001 | -0.0011 |
| musique | zero-shot | 2417 | 0.5013 | 0.5127 | +0.0114 [+0.0065, +0.0164] | GAIN | 0.4734 | +0.0074 | -0.0037 |
| hotpotqa | in-domain | 7405 | 0.9035 | 0.9086 | +0.0051 [+0.0030, +0.0074] | WITHIN | 0.6846 | +0.0084 | +0.0032 |
| 2wiki | in-domain | 12576 | 0.9377 | 0.9373 | -0.0004 [-0.0017, +0.0011] | WITHIN | 0.6079 | -0.0024 | -0.0072 |
| webqsp | zero-shot | 1503 | 0.3768 | 0.3962 | +0.0194 [+0.0092, +0.0296] | GAIN | 0.0535 | +0.0106 | +0.0120 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8014 | +0.1477 [+0.1422, +0.1529] | GAIN | 0.0047 | +0.1466 | +0.2799 |
| squad | in-domain | 11873 | 0.9100 | 0.9126 | +0.0026 [+0.0005, +0.0048] | WITHIN | 0.9054 | +0.0026 | -0.0033 |
| musique | zero-shot | 2417 | 0.2696 | 0.5127 | +0.2431 [+0.2313, +0.2552] | GAIN | 0.4734 | +0.1237 | +0.4162 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9086 | +0.0072 [+0.0041, +0.0102] | WITHIN | 0.6846 | +0.0097 | -0.0216 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9373 | +0.0641 [+0.0609, +0.0673] | GAIN | 0.6079 | +0.1322 | +0.0638 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3962 | +0.2184 [+0.1967, +0.2419] | GAIN | 0.0535 | +0.1816 | +0.1065 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
