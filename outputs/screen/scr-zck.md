# Screen scr-zck (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrct (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7782 | 0.7783 | +0.0000 [-0.0010, +0.0010] | WITHIN | 0.0047 | +0.0005 | -0.0008 |
| squad | in-domain | 11873 | 0.9105 | 0.9116 | +0.0011 [-0.0008, +0.0028] | WITHIN | 0.9054 | +0.0011 | -0.0025 |
| musique | zero-shot | 2417 | 0.5235 | 0.5275 | +0.0039 [-0.0009, +0.0088] | WITHIN | 0.4734 | +0.0033 | -0.0062 |
| hotpotqa | in-domain | 7405 | 0.9041 | 0.9041 | +0.0000 [-0.0022, +0.0022] | WITHIN | 0.6846 | -0.0009 | -0.0086 |
| 2wiki | in-domain | 12576 | 0.8733 | 0.8692 | -0.0041 [-0.0059, -0.0023] | WITHIN | 0.6079 | -0.0068 | -0.0153 |
| webqsp | zero-shot | 1503 | 0.3384 | 0.3544 | +0.0160 [+0.0042, +0.0270] | GAIN | 0.0535 | +0.0146 | +0.0333 |

## Against outputs\screen\fits\scr-zrm (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7789 | 0.7783 | -0.0007 [-0.0018, +0.0004] | WITHIN | 0.0047 | -0.0002 | -0.0025 |
| squad | in-domain | 11873 | 0.9111 | 0.9116 | +0.0005 [-0.0013, +0.0024] | WITHIN | 0.9054 | +0.0005 | -0.0010 |
| musique | zero-shot | 2417 | 0.5231 | 0.5275 | +0.0044 [-0.0007, +0.0093] | WITHIN | 0.4734 | +0.0004 | -0.0157 |
| hotpotqa | in-domain | 7405 | 0.9040 | 0.9041 | +0.0001 [-0.0022, +0.0024] | WITHIN | 0.6846 | +0.0014 | +0.0032 |
| 2wiki | in-domain | 12576 | 0.8709 | 0.8692 | -0.0017 [-0.0035, -0.0000] | WITHIN | 0.6079 | -0.0014 | -0.0053 |
| webqsp | zero-shot | 1503 | 0.3128 | 0.3544 | +0.0416 [+0.0252, +0.0574] | GAIN | 0.0535 | +0.0346 | -0.0200 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7783 | +0.1245 [+0.1194, +0.1298] | GAIN | 0.0047 | +0.1271 | +0.2497 |
| squad | in-domain | 11873 | 0.9100 | 0.9116 | +0.0016 [-0.0004, +0.0036] | WITHIN | 0.9054 | +0.0016 | -0.0027 |
| musique | zero-shot | 2417 | 0.2696 | 0.5275 | +0.2579 [+0.2463, +0.2695] | GAIN | 0.4734 | +0.1502 | +0.3686 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9041 | +0.0026 [+0.0001, +0.0050] | WITHIN | 0.6846 | +0.0043 | +0.0020 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8692 | -0.0040 [-0.0061, -0.0020] | WITHIN | 0.6079 | -0.0080 | -0.0075 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3544 | +0.1766 [+0.1546, +0.1994] | GAIN | 0.0535 | +0.1444 | +0.0858 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
