# Screen scr-wqk-zspk-s2-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zspk-s0-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3384 | 0.3417 | +0.0034 [-0.0024, +0.0092] | WITHIN | 0.0047 | -0.0028 | +0.0093 |
| squad | in-domain | 11873 | 0.9114 | 0.9114 | +0.0000 [-0.0021, +0.0021] | WITHIN | 0.9054 | +0.0000 | +0.0027 |
| musique | in-domain | 2417 | 0.5794 | 0.5850 | +0.0057 [+0.0006, +0.0110] | WITHIN | 0.4734 | +0.0058 | +0.0021 |
| hotpotqa | zero-shot | 7405 | 0.8496 | 0.8460 | -0.0036 [-0.0070, -0.0003] | WITHIN | 0.6846 | -0.0024 | -0.0190 |
| 2wiki | in-domain | 12576 | 0.9330 | 0.9353 | +0.0023 [+0.0007, +0.0040] | WITHIN | 0.6079 | +0.0066 | -0.0037 |
| webqsp | in-domain | 1503 | 0.6153 | 0.6213 | +0.0060 [-0.0046, +0.0170] | WITHIN | 0.0535 | +0.0106 | +0.0106 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
