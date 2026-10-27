# Screen scr-a5cp-zspkc-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1gb-zspk-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8011 | 0.8004 | -0.0007 [-0.0018, +0.0004] | WITHIN | 0.0047 | +0.0007 | -0.0008 |
| squad | in-domain | 11873 | 0.9127 | 0.9127 | +0.0001 [-0.0019, +0.0020] | WITHIN | 0.9054 | +0.0001 | -0.0012 |
| musique | in-domain | 2417 | 0.5662 | 0.5784 | +0.0122 [+0.0067, +0.0180] | GAIN | 0.4734 | +0.0091 | -0.0029 |
| hotpotqa | zero-shot | 7405 | 0.8483 | 0.8528 | +0.0045 [+0.0018, +0.0072] | WITHIN | 0.6846 | +0.0101 | -0.0090 |
| 2wiki | in-domain | 12576 | 0.9326 | 0.9370 | +0.0044 [+0.0029, +0.0060] | WITHIN | 0.6079 | +0.0119 | -0.0054 |
| webqsp | zero-shot | 1503 | 0.2808 | 0.3264 | +0.0455 [+0.0314, +0.0599] | GAIN | 0.0535 | +0.0419 | +0.0373 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8004 | +0.1464 [+0.1409, +0.1516] | GAIN | 0.0047 | +0.1524 | +0.2806 |
| squad | in-domain | 11873 | 0.9116 | 0.9127 | +0.0011 [-0.0013, +0.0035] | WITHIN | 0.9054 | +0.0011 | +0.0004 |
| musique | in-domain | 2417 | 0.5550 | 0.5784 | +0.0234 [+0.0154, +0.0316] | GAIN | 0.4734 | +0.0240 | +0.0451 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8528 | -0.0057 [-0.0105, -0.0012] | WITHIN | 0.6846 | -0.0111 | -0.0475 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9370 | +0.0698 [+0.0665, +0.0732] | GAIN | 0.6079 | +0.1433 | +0.0656 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3264 | +0.2064 [+0.1807, +0.2329] | GAIN | 0.0535 | +0.1603 | +0.1251 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
