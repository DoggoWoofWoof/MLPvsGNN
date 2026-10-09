# Screen scr-zkind-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrct-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7793 | 0.7779 | -0.0014 [-0.0024, -0.0003] | WITHIN | 0.0047 | -0.0009 | -0.0041 |
| squad | in-domain | 11873 | 0.9112 | 0.9140 | +0.0028 [+0.0008, +0.0049] | WITHIN | 0.9054 | +0.0028 | -0.0019 |
| musique | in-domain | 2417 | 0.5645 | 0.5677 | +0.0033 [-0.0029, +0.0098] | WITHIN | 0.4734 | +0.0153 | +0.0012 |
| hotpotqa | zero-shot | 7405 | 0.8451 | 0.8344 | -0.0107 [-0.0142, -0.0074] | LOSS | 0.6846 | -0.0238 | -0.0020 |
| 2wiki | in-domain | 12576 | 0.8731 | 0.8687 | -0.0044 [-0.0064, -0.0024] | WITHIN | 0.6079 | -0.0132 | +0.0015 |
| webqsp | zero-shot | 1503 | 0.2865 | 0.3007 | +0.0142 [-0.0037, +0.0312] | WITHIN | 0.0535 | +0.0047 | +0.0652 |

## Against outputs\screen\fits\scr-zrm-hp (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7801 | 0.7779 | -0.0022 [-0.0033, -0.0010] | WITHIN | 0.0047 | -0.0013 | -0.0044 |
| squad | in-domain | 11873 | 0.9116 | 0.9140 | +0.0024 [+0.0004, +0.0045] | WITHIN | 0.9054 | +0.0024 | -0.0008 |
| musique | in-domain | 2417 | 0.5600 | 0.5677 | +0.0078 [+0.0019, +0.0139] | GAIN | 0.4734 | +0.0132 | -0.0108 |
| hotpotqa | zero-shot | 7405 | 0.8457 | 0.8344 | -0.0113 [-0.0149, -0.0078] | LOSS | 0.6846 | -0.0246 | -0.0063 |
| 2wiki | in-domain | 12576 | 0.8681 | 0.8687 | +0.0007 [-0.0014, +0.0027] | WITHIN | 0.6079 | +0.0014 | -0.0009 |
| webqsp | zero-shot | 1503 | 0.2416 | 0.3007 | +0.0591 [+0.0444, +0.0741] | GAIN | 0.0535 | +0.0479 | +0.0492 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7779 | +0.1239 [+0.1186, +0.1291] | GAIN | 0.0047 | +0.1325 | +0.2512 |
| squad | in-domain | 11873 | 0.9116 | 0.9140 | +0.0024 [+0.0003, +0.0046] | WITHIN | 0.9054 | +0.0024 | +0.0012 |
| musique | in-domain | 2417 | 0.5550 | 0.5677 | +0.0127 [+0.0065, +0.0194] | GAIN | 0.4734 | +0.0310 | +0.0017 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8344 | -0.0242 [-0.0282, -0.0199] | LOSS | 0.6846 | -0.0481 | -0.0020 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8687 | +0.0015 [-0.0009, +0.0040] | WITHIN | 0.6079 | +0.0041 | -0.0073 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3007 | +0.1807 [+0.1562, +0.2048] | GAIN | 0.0535 | +0.1411 | +0.1124 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
