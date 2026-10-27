# Screen scr-p1gb-zspk (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8004 | +0.1466 [+0.1413, +0.1518] | GAIN | 0.0047 | +0.1456 | +0.2789 |
| squad | in-domain | 11873 | 0.9100 | 0.9129 | +0.0029 [+0.0008, +0.0052] | WITHIN | 0.9054 | +0.0029 | +0.0004 |
| musique | zero-shot | 2417 | 0.2696 | 0.5169 | +0.2473 [+0.2359, +0.2590] | GAIN | 0.4734 | +0.1324 | +0.4175 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9086 | +0.0072 [+0.0039, +0.0104] | WITHIN | 0.6846 | +0.0096 | -0.0158 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9344 | +0.0612 [+0.0580, +0.0644] | GAIN | 0.6079 | +0.1236 | +0.0711 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3354 | +0.1575 [+0.1332, +0.1824] | GAIN | 0.0535 | +0.1211 | +0.1238 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
