# Screen J5 (p@swa, s1eval carves)

## Against outputs\full_zrct\fits\J5 (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7796 | 0.7797 | +0.0001 [-0.0010, +0.0011] | WITHIN | 0.0047 | +0.0005 | -0.0058 |
| squad | in-domain | 11873 | 0.9107 | 0.9115 | +0.0008 [-0.0012, +0.0026] | WITHIN | 0.9054 | +0.0008 | +0.0002 |
| musique | in-domain | 2417 | 0.5669 | 0.5699 | +0.0031 [-0.0023, +0.0086] | WITHIN | 0.4734 | +0.0004 | +0.0008 |
| hotpotqa | in-domain | 7405 | 0.9049 | 0.8986 | -0.0063 [-0.0090, -0.0039] | WITHIN | 0.6846 | -0.0100 | +0.0078 |
| 2wiki | in-domain | 12576 | 0.8697 | 0.8660 | -0.0037 [-0.0057, -0.0017] | WITHIN | 0.6079 | -0.0026 | -0.0002 |
| webqsp | zero-shot | 1503 | 0.3071 | 0.2930 | -0.0141 [-0.0329, +0.0040] | WITHIN | 0.0535 | -0.0113 | +0.0639 |

## Against outputs\full_zrm\fits\J5 (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7788 | 0.7797 | +0.0008 [-0.0002, +0.0019] | WITHIN | 0.0047 | +0.0007 | -0.0036 |
| squad | in-domain | 11873 | 0.9108 | 0.9115 | +0.0007 [-0.0013, +0.0026] | WITHIN | 0.9054 | +0.0007 | +0.0005 |
| musique | in-domain | 2417 | 0.5665 | 0.5699 | +0.0034 [-0.0023, +0.0091] | WITHIN | 0.4734 | +0.0033 | +0.0000 |
| hotpotqa | in-domain | 7405 | 0.8997 | 0.8986 | -0.0011 [-0.0037, +0.0016] | WITHIN | 0.6846 | -0.0009 | +0.0003 |
| 2wiki | in-domain | 12576 | 0.8705 | 0.8660 | -0.0046 [-0.0066, -0.0026] | WITHIN | 0.6079 | -0.0050 | -0.0111 |
| webqsp | zero-shot | 1503 | 0.2558 | 0.2930 | +0.0372 [+0.0222, +0.0524] | GAIN | 0.0535 | +0.0293 | +0.0466 |

## Against outputs\step1\fits\J5 (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6443 | 0.7797 | +0.1354 [+0.1300, +0.1410] | GAIN | 0.0047 | +0.1416 | +0.2520 |
| squad | in-domain | 11873 | 0.9095 | 0.9115 | +0.0019 [-0.0005, +0.0043] | WITHIN | 0.9054 | +0.0019 | -0.0008 |
| musique | in-domain | 2417 | 0.5609 | 0.5699 | +0.0091 [+0.0026, +0.0154] | GAIN | 0.4734 | +0.0137 | +0.0066 |
| hotpotqa | in-domain | 7405 | 0.9013 | 0.8986 | -0.0027 [-0.0057, +0.0003] | WITHIN | 0.6846 | -0.0028 | +0.0081 |
| 2wiki | in-domain | 12576 | 0.8694 | 0.8660 | -0.0035 [-0.0056, -0.0014] | WITHIN | 0.6079 | -0.0051 | -0.0123 |
| webqsp | zero-shot | 1503 | 0.1167 | 0.2930 | +0.1763 [+0.1518, +0.2000] | GAIN | 0.0535 | +0.1377 | +0.1191 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
