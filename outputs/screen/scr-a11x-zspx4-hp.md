# Screen scr-a11x-zspx4-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g5-zspx-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8039 | 0.8050 | +0.0011 [+0.0001, +0.0019] | WITHIN | 0.0047 | +0.0005 | +0.0044 |
| squad | in-domain | 11873 | 0.9129 | 0.9131 | +0.0002 [-0.0018, +0.0022] | WITHIN | 0.9054 | +0.0002 | -0.0012 |
| musique | in-domain | 2417 | 0.5901 | 0.5881 | -0.0020 [-0.0069, +0.0028] | WITHIN | 0.4734 | -0.0058 | -0.0050 |
| hotpotqa | zero-shot | 7405 | 0.8547 | 0.8530 | -0.0017 [-0.0043, +0.0011] | WITHIN | 0.6846 | -0.0024 | -0.0014 |
| 2wiki | in-domain | 12576 | 0.9312 | 0.9327 | +0.0015 [+0.0001, +0.0029] | WITHIN | 0.6079 | +0.0025 | -0.0059 |
| webqsp | zero-shot | 1503 | 0.2914 | 0.2770 | -0.0145 [-0.0229, -0.0061] | LOSS | 0.0535 | -0.0093 | -0.0067 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8050 | +0.1510 [+0.1454, +0.1565] | GAIN | 0.0047 | +0.1553 | +0.2969 |
| squad | in-domain | 11873 | 0.9116 | 0.9131 | +0.0014 [-0.0010, +0.0039] | WITHIN | 0.9054 | +0.0014 | +0.0007 |
| musique | in-domain | 2417 | 0.5550 | 0.5881 | +0.0331 [+0.0255, +0.0406] | GAIN | 0.4734 | +0.0364 | +0.0501 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8530 | -0.0055 [-0.0107, -0.0007] | WITHIN | 0.6846 | -0.0090 | -0.0251 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9327 | +0.0655 [+0.0620, +0.0691] | GAIN | 0.6079 | +0.1336 | +0.0689 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2770 | +0.1570 [+0.1332, +0.1809] | GAIN | 0.0535 | +0.1178 | +0.0845 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
