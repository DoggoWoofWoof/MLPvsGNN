# Screen scr-a5cp-zrckpc (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1gb-zrckp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7993 | 0.7997 | +0.0004 [-0.0006, +0.0015] | WITHIN | 0.0047 | +0.0006 | +0.0027 |
| squad | in-domain | 11873 | 0.9127 | 0.9117 | -0.0009 [-0.0027, +0.0009] | WITHIN | 0.9054 | -0.0009 | +0.0013 |
| musique | zero-shot | 2417 | 0.5013 | 0.5111 | +0.0098 [+0.0051, +0.0147] | GAIN | 0.4734 | +0.0108 | -0.0070 |
| hotpotqa | in-domain | 7405 | 0.9035 | 0.9061 | +0.0026 [+0.0003, +0.0050] | WITHIN | 0.6846 | +0.0047 | +0.0088 |
| 2wiki | in-domain | 12576 | 0.9377 | 0.9332 | -0.0045 [-0.0060, -0.0030] | WITHIN | 0.6079 | -0.0134 | +0.0007 |
| webqsp | zero-shot | 1503 | 0.3768 | 0.3928 | +0.0160 [+0.0039, +0.0280] | GAIN | 0.0535 | +0.0086 | +0.0040 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7997 | +0.1459 [+0.1405, +0.1510] | GAIN | 0.0047 | +0.1453 | +0.2710 |
| squad | in-domain | 11873 | 0.9100 | 0.9117 | +0.0018 [-0.0004, +0.0040] | WITHIN | 0.9054 | +0.0018 | -0.0009 |
| musique | zero-shot | 2417 | 0.2696 | 0.5111 | +0.2416 [+0.2299, +0.2529] | GAIN | 0.4734 | +0.1270 | +0.4129 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9061 | +0.0047 [+0.0013, +0.0079] | WITHIN | 0.6846 | +0.0061 | -0.0161 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9332 | +0.0600 [+0.0569, +0.0632] | GAIN | 0.6079 | +0.1212 | +0.0717 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3928 | +0.2150 [+0.1917, +0.2398] | GAIN | 0.0535 | +0.1796 | +0.0985 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
