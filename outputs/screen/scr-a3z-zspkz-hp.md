# Screen scr-a3z-zspkz-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3-zspk-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8040 | 0.8015 | -0.0025 [-0.0035, -0.0015] | WITHIN | 0.0047 | -0.0018 | -0.0106 |
| squad | in-domain | 11873 | 0.9135 | 0.9132 | -0.0003 [-0.0024, +0.0018] | WITHIN | 0.9054 | -0.0003 | -0.0034 |
| musique | in-domain | 2417 | 0.5754 | 0.5714 | -0.0040 [-0.0091, +0.0009] | WITHIN | 0.4734 | -0.0087 | -0.0025 |
| hotpotqa | zero-shot | 7405 | 0.8499 | 0.8427 | -0.0072 [-0.0103, -0.0040] | WITHIN | 0.6846 | -0.0077 | -0.0190 |
| 2wiki | in-domain | 12576 | 0.9333 | 0.9330 | -0.0003 [-0.0019, +0.0014] | WITHIN | 0.6079 | +0.0031 | +0.0020 |
| webqsp | zero-shot | 1503 | 0.2680 | 0.2848 | +0.0168 [+0.0068, +0.0268] | GAIN | 0.0535 | +0.0113 | +0.0007 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8015 | +0.1475 [+0.1417, +0.1531] | GAIN | 0.0047 | +0.1520 | +0.2792 |
| squad | in-domain | 11873 | 0.9116 | 0.9132 | +0.0016 [-0.0008, +0.0039] | WITHIN | 0.9054 | +0.0016 | -0.0010 |
| musique | in-domain | 2417 | 0.5550 | 0.5714 | +0.0163 [+0.0088, +0.0239] | GAIN | 0.4734 | +0.0153 | +0.0476 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8427 | -0.0158 [-0.0208, -0.0108] | LOSS | 0.6846 | -0.0248 | -0.0586 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9330 | +0.0658 [+0.0623, +0.0693] | GAIN | 0.6079 | +0.1372 | +0.0755 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2848 | +0.1648 [+0.1412, +0.1888] | GAIN | 0.0535 | +0.1218 | +0.0918 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
