# Screen scr-p5c5-zrckpo5 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3p-zrckp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8019 | 0.8013 | -0.0006 [-0.0016, +0.0003] | WITHIN | 0.0047 | -0.0001 | -0.0035 |
| squad | in-domain | 11873 | 0.9116 | 0.9113 | -0.0003 [-0.0019, +0.0012] | WITHIN | 0.9054 | -0.0003 | -0.0026 |
| musique | zero-shot | 2417 | 0.5121 | 0.5132 | +0.0011 [-0.0032, +0.0056] | WITHIN | 0.4734 | +0.0012 | +0.0137 |
| hotpotqa | in-domain | 7405 | 0.9065 | 0.9069 | +0.0003 [-0.0016, +0.0022] | WITHIN | 0.6846 | +0.0018 | -0.0074 |
| 2wiki | in-domain | 12576 | 0.9347 | 0.9365 | +0.0018 [+0.0004, +0.0032] | WITHIN | 0.6079 | +0.0045 | -0.0038 |
| webqsp | zero-shot | 1503 | 0.3627 | 0.3551 | -0.0076 [-0.0172, +0.0016] | WITHIN | 0.0535 | -0.0133 | +0.0033 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8013 | +0.1475 [+0.1419, +0.1527] | GAIN | 0.0047 | +0.1453 | +0.2777 |
| squad | in-domain | 11873 | 0.9100 | 0.9113 | +0.0013 [-0.0008, +0.0035] | WITHIN | 0.9054 | +0.0013 | -0.0037 |
| musique | zero-shot | 2417 | 0.2696 | 0.5132 | +0.2436 [+0.2317, +0.2557] | GAIN | 0.4734 | +0.1221 | +0.4270 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9069 | +0.0054 [+0.0022, +0.0086] | WITHIN | 0.6846 | +0.0059 | -0.0149 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9365 | +0.0633 [+0.0603, +0.0665] | GAIN | 0.6079 | +0.1307 | +0.0689 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3551 | +0.1773 [+0.1566, +0.1994] | GAIN | 0.0535 | +0.1450 | +0.0758 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
