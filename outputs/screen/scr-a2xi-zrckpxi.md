# Screen scr-a2xi-zrckpxi (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3p-zrckp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8019 | 0.8044 | +0.0025 [+0.0013, +0.0035] | WITHIN | 0.0047 | +0.0040 | +0.0082 |
| squad | in-domain | 11873 | 0.9116 | 0.9102 | -0.0014 [-0.0037, +0.0008] | WITHIN | 0.9054 | -0.0014 | -0.0084 |
| musique | zero-shot | 2417 | 0.5121 | 0.5096 | -0.0026 [-0.0085, +0.0036] | WITHIN | 0.4734 | +0.0062 | -0.0128 |
| hotpotqa | in-domain | 7405 | 0.9065 | 0.9017 | -0.0049 [-0.0075, -0.0022] | WITHIN | 0.6846 | -0.0055 | -0.0197 |
| 2wiki | in-domain | 12576 | 0.9347 | 0.9381 | +0.0034 [+0.0017, +0.0051] | WITHIN | 0.6079 | +0.0083 | +0.0044 |
| webqsp | zero-shot | 1503 | 0.3627 | 0.3859 | +0.0231 [+0.0119, +0.0343] | GAIN | 0.0535 | +0.0146 | +0.0386 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8044 | +0.1506 [+0.1450, +0.1560] | GAIN | 0.0047 | +0.1494 | +0.2893 |
| squad | in-domain | 11873 | 0.9100 | 0.9102 | +0.0003 [-0.0025, +0.0028] | WITHIN | 0.9054 | +0.0003 | -0.0095 |
| musique | zero-shot | 2417 | 0.2696 | 0.5096 | +0.2400 [+0.2275, +0.2525] | GAIN | 0.4734 | +0.1270 | +0.4005 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9017 | +0.0002 [-0.0034, +0.0037] | WITHIN | 0.6846 | -0.0014 | -0.0271 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9381 | +0.0649 [+0.0617, +0.0682] | GAIN | 0.6079 | +0.1345 | +0.0771 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3859 | +0.2080 [+0.1860, +0.2312] | GAIN | 0.0535 | +0.1730 | +0.1111 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
