# Screen scr-a12x-zspxgs-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g5-zspx-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8039 | 0.8025 | -0.0014 [-0.0023, -0.0006] | WITHIN | 0.0047 | -0.0015 | -0.0085 |
| squad | in-domain | 11873 | 0.9129 | 0.9128 | -0.0001 [-0.0019, +0.0019] | WITHIN | 0.9054 | -0.0001 | -0.0005 |
| musique | in-domain | 2417 | 0.5901 | 0.5851 | -0.0050 [-0.0102, +0.0001] | WITHIN | 0.4734 | -0.0095 | -0.0066 |
| hotpotqa | zero-shot | 7405 | 0.8547 | 0.8514 | -0.0033 [-0.0061, -0.0004] | WITHIN | 0.6846 | -0.0069 | +0.0072 |
| 2wiki | in-domain | 12576 | 0.9312 | 0.9364 | +0.0051 [+0.0037, +0.0066] | WITHIN | 0.6079 | +0.0117 | -0.0010 |
| webqsp | zero-shot | 1503 | 0.2914 | 0.2821 | -0.0094 [-0.0174, -0.0012] | LOSS | 0.0535 | -0.0073 | +0.0027 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8025 | +0.1485 [+0.1428, +0.1541] | GAIN | 0.0047 | +0.1533 | +0.2840 |
| squad | in-domain | 11873 | 0.9116 | 0.9128 | +0.0012 [-0.0012, +0.0035] | WITHIN | 0.9054 | +0.0012 | +0.0013 |
| musique | in-domain | 2417 | 0.5550 | 0.5851 | +0.0301 [+0.0227, +0.0377] | GAIN | 0.4734 | +0.0327 | +0.0484 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8514 | -0.0072 [-0.0120, -0.0022] | WITHIN | 0.6846 | -0.0135 | -0.0166 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9364 | +0.0691 [+0.0658, +0.0727] | GAIN | 0.6079 | +0.1428 | +0.0737 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2821 | +0.1621 [+0.1384, +0.1870] | GAIN | 0.0535 | +0.1198 | +0.0938 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
