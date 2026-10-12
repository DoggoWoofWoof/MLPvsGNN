# Screen scr-u1c-zsp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zspg (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7794 | 0.7835 | +0.0041 [+0.0021, +0.0061] | WITHIN | 0.0047 | +0.0044 | -0.0185 |
| squad | in-domain | 11873 | 0.9125 | 0.9132 | +0.0008 [-0.0014, +0.0029] | WITHIN | 0.9054 | +0.0008 | +0.0029 |
| musique | zero-shot | 2417 | 0.5111 | 0.4922 | -0.0189 [-0.0280, -0.0103] | LOSS | 0.4734 | -0.0248 | -0.1465 |
| hotpotqa | in-domain | 7405 | 0.9065 | 0.8431 | -0.0634 [-0.0677, -0.0589] | LOSS | 0.6846 | -0.1248 | -0.0113 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.8612 | -0.0112 [-0.0145, -0.0079] | LOSS | 0.6079 | -0.0285 | +0.0084 |
| webqsp | zero-shot | 1503 | 0.3245 | 0.2683 | -0.0562 [-0.0719, -0.0406] | LOSS | 0.0535 | -0.0446 | -0.0459 |

## Against outputs\step1\fits\L-musique (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7835 | +0.1298 [+0.1245, +0.1351] | GAIN | 0.0047 | +0.1319 | +0.2351 |
| squad | in-domain | 11873 | 0.9100 | 0.9132 | +0.0033 [+0.0011, +0.0056] | WITHIN | 0.9054 | +0.0033 | -0.0003 |
| musique | zero-shot | 2417 | 0.2696 | 0.4922 | +0.2227 [+0.2102, +0.2344] | GAIN | 0.4734 | +0.1121 | +0.2089 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.8431 | -0.0583 [-0.0629, -0.0538] | LOSS | 0.6846 | -0.1167 | +0.0018 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8612 | -0.0120 [-0.0152, -0.0088] | LOSS | 0.6079 | -0.0263 | +0.0062 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.2683 | +0.0904 [+0.0680, +0.1134] | GAIN | 0.0535 | +0.0719 | +0.0685 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
