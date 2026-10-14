# Screen scr-p1c3p-zrckp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrct (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7782 | 0.8019 | +0.0237 [+0.0207, +0.0265] | GAIN | 0.0047 | +0.0188 | +0.0307 |
| squad | in-domain | 11873 | 0.9105 | 0.9116 | +0.0012 [-0.0008, +0.0033] | WITHIN | 0.9054 | +0.0012 | -0.0009 |
| musique | zero-shot | 2417 | 0.5235 | 0.5121 | -0.0114 [-0.0198, -0.0033] | LOSS | 0.4734 | -0.0261 | +0.0385 |
| hotpotqa | in-domain | 7405 | 0.9041 | 0.9065 | +0.0025 [-0.0008, +0.0059] | WITHIN | 0.6846 | -0.0011 | -0.0181 |
| 2wiki | in-domain | 12576 | 0.8733 | 0.9347 | +0.0614 [+0.0582, +0.0647] | GAIN | 0.6079 | +0.1275 | +0.0649 |
| webqsp | zero-shot | 1503 | 0.3384 | 0.3627 | +0.0243 [+0.0088, +0.0400] | GAIN | 0.0535 | +0.0286 | +0.0200 |

## Against outputs\screen\fits\scr-p1c3-zrck (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8007 | 0.8019 | +0.0012 [+0.0002, +0.0021] | WITHIN | 0.0047 | -0.0006 | +0.0040 |
| squad | in-domain | 11873 | 0.9117 | 0.9116 | -0.0001 [-0.0019, +0.0017] | WITHIN | 0.9054 | -0.0001 | +0.0027 |
| musique | zero-shot | 2417 | 0.5059 | 0.5121 | +0.0062 [-0.0001, +0.0123] | WITHIN | 0.4734 | +0.0070 | -0.0149 |
| hotpotqa | in-domain | 7405 | 0.9063 | 0.9065 | +0.0003 [-0.0017, +0.0022] | WITHIN | 0.6846 | -0.0008 | +0.0005 |
| 2wiki | in-domain | 12576 | 0.9365 | 0.9347 | -0.0018 [-0.0033, -0.0004] | WITHIN | 0.6079 | -0.0041 | +0.0024 |
| webqsp | zero-shot | 1503 | 0.3756 | 0.3627 | -0.0129 [-0.0230, -0.0028] | LOSS | 0.0535 | -0.0020 | -0.0047 |

## Against outputs\screen\fits\scr-p1c2-zrc (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7994 | 0.8019 | +0.0026 [+0.0013, +0.0039] | WITHIN | 0.0047 | +0.0002 | +0.0067 |
| squad | in-domain | 11873 | 0.9123 | 0.9116 | -0.0007 [-0.0027, +0.0013] | WITHIN | 0.9054 | -0.0007 | +0.0043 |
| musique | zero-shot | 2417 | 0.4789 | 0.5121 | +0.0333 [+0.0245, +0.0428] | GAIN | 0.4734 | +0.0265 | +0.0571 |
| hotpotqa | in-domain | 7405 | 0.9036 | 0.9065 | +0.0029 [+0.0001, +0.0056] | WITHIN | 0.6846 | +0.0019 | -0.0090 |
| 2wiki | in-domain | 12576 | 0.9128 | 0.9347 | +0.0219 [+0.0198, +0.0242] | GAIN | 0.6079 | +0.0509 | +0.0421 |
| webqsp | zero-shot | 1503 | 0.3466 | 0.3627 | +0.0161 [+0.0052, +0.0276] | GAIN | 0.0535 | +0.0206 | +0.0067 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8019 | +0.1482 [+0.1426, +0.1535] | GAIN | 0.0047 | +0.1454 | +0.2811 |
| squad | in-domain | 11873 | 0.9100 | 0.9116 | +0.0017 [-0.0006, +0.0039] | WITHIN | 0.9054 | +0.0017 | -0.0011 |
| musique | zero-shot | 2417 | 0.2696 | 0.5121 | +0.2426 [+0.2305, +0.2546] | GAIN | 0.4734 | +0.1208 | +0.4133 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9065 | +0.0051 [+0.0017, +0.0084] | WITHIN | 0.6846 | +0.0042 | -0.0074 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9347 | +0.0615 [+0.0582, +0.0647] | GAIN | 0.6079 | +0.1263 | +0.0727 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3627 | +0.1849 [+0.1639, +0.2079] | GAIN | 0.0535 | +0.1583 | +0.0725 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
