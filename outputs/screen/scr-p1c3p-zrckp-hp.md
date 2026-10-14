# Screen scr-p1c3p-zrckp-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrct-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7793 | 0.8044 | +0.0251 [+0.0222, +0.0279] | GAIN | 0.0047 | +0.0204 | +0.0331 |
| squad | in-domain | 11873 | 0.9112 | 0.9127 | +0.0015 [-0.0006, +0.0035] | WITHIN | 0.9054 | +0.0015 | -0.0007 |
| musique | in-domain | 2417 | 0.5645 | 0.5745 | +0.0100 [+0.0021, +0.0176] | GAIN | 0.4734 | +0.0025 | +0.0327 |
| hotpotqa | zero-shot | 7405 | 0.8451 | 0.8513 | +0.0062 [+0.0020, +0.0106] | WITHIN | 0.6846 | +0.0092 | -0.0344 |
| 2wiki | in-domain | 12576 | 0.8731 | 0.9350 | +0.0619 [+0.0587, +0.0652] | GAIN | 0.6079 | +0.1236 | +0.0734 |
| webqsp | zero-shot | 1503 | 0.2865 | 0.3475 | +0.0609 [+0.0430, +0.0776] | GAIN | 0.0535 | +0.0486 | +0.0213 |

## Against outputs\screen\fits\scr-p1c3-zrck-hp (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8052 | 0.8044 | -0.0008 [-0.0016, +0.0000] | WITHIN | 0.0047 | -0.0013 | -0.0050 |
| squad | in-domain | 11873 | 0.9122 | 0.9127 | +0.0006 [-0.0014, +0.0024] | WITHIN | 0.9054 | +0.0006 | +0.0016 |
| musique | in-domain | 2417 | 0.5694 | 0.5745 | +0.0050 [-0.0012, +0.0110] | WITHIN | 0.4734 | +0.0017 | -0.0219 |
| hotpotqa | zero-shot | 7405 | 0.8564 | 0.8513 | -0.0051 [-0.0087, -0.0016] | WITHIN | 0.6846 | -0.0113 | +0.0080 |
| 2wiki | in-domain | 12576 | 0.9128 | 0.9350 | +0.0222 [+0.0200, +0.0243] | GAIN | 0.6079 | +0.0647 | -0.0031 |
| webqsp | zero-shot | 1503 | 0.3686 | 0.3475 | -0.0211 [-0.0319, -0.0105] | LOSS | 0.0535 | -0.0180 | -0.0053 |

## Against outputs\screen\fits\scr-p1c2-zrc-hp (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8002 | 0.8044 | +0.0042 [+0.0029, +0.0055] | WITHIN | 0.0047 | +0.0020 | +0.0158 |
| squad | in-domain | 11873 | 0.9134 | 0.9127 | -0.0007 [-0.0027, +0.0013] | WITHIN | 0.9054 | -0.0007 | -0.0004 |
| musique | in-domain | 2417 | 0.5651 | 0.5745 | +0.0094 [+0.0024, +0.0167] | GAIN | 0.4734 | +0.0025 | +0.0157 |
| hotpotqa | zero-shot | 7405 | 0.8363 | 0.8513 | +0.0151 [+0.0114, +0.0187] | GAIN | 0.6846 | +0.0251 | -0.0321 |
| 2wiki | in-domain | 12576 | 0.9125 | 0.9350 | +0.0225 [+0.0202, +0.0247] | GAIN | 0.6079 | +0.0492 | +0.0306 |
| webqsp | zero-shot | 1503 | 0.3142 | 0.3475 | +0.0333 [+0.0196, +0.0464] | GAIN | 0.0535 | +0.0286 | +0.0173 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8044 | +0.1504 [+0.1447, +0.1558] | GAIN | 0.0047 | +0.1539 | +0.2884 |
| squad | in-domain | 11873 | 0.9116 | 0.9127 | +0.0011 [-0.0010, +0.0033] | WITHIN | 0.9054 | +0.0011 | +0.0024 |
| musique | in-domain | 2417 | 0.5550 | 0.5745 | +0.0194 [+0.0122, +0.0269] | GAIN | 0.4734 | +0.0182 | +0.0331 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8513 | -0.0072 [-0.0120, -0.0026] | WITHIN | 0.6846 | -0.0151 | -0.0344 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9350 | +0.0678 [+0.0645, +0.0712] | GAIN | 0.6079 | +0.1410 | +0.0646 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3475 | +0.2275 [+0.2033, +0.2523] | GAIN | 0.0535 | +0.1850 | +0.0685 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
