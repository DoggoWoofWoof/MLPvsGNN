# Screen scr-rel-hp (p@swa, s1eval carves)

## Against outputs\step1\fits\L-hotpotqa (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7217 | +0.0677 [+0.0633, +0.0719] | GAIN | 0.0047 | +0.0621 | +0.1392 |
| squad | in-domain | 11873 | 0.9116 | 0.9105 | -0.0012 [-0.0032, +0.0009] | WITHIN | 0.9054 | -0.0012 | +0.0003 |
| musique | in-domain | 2417 | 0.5550 | 0.5603 | +0.0053 [-0.0006, +0.0112] | WITHIN | 0.4734 | +0.0120 | +0.0116 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8453 | -0.0132 [-0.0171, -0.0095] | LOSS | 0.6846 | -0.0258 | +0.0003 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8688 | +0.0015 [-0.0006, +0.0036] | WITHIN | 0.6079 | +0.0045 | -0.0085 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.1902 | +0.0702 [+0.0514, +0.0893] | GAIN | 0.0535 | +0.0546 | +0.0393 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
