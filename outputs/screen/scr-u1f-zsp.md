# Screen scr-u1f-zsp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zspg (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7794 | 0.7834 | +0.0040 [+0.0019, +0.0061] | WITHIN | 0.0047 | +0.0046 | -0.0154 |
| squad | in-domain | 11873 | 0.9125 | 0.9132 | +0.0008 [-0.0014, +0.0027] | WITHIN | 0.9054 | +0.0008 | +0.0038 |
| musique | zero-shot | 2417 | 0.5111 | 0.4978 | -0.0133 [-0.0214, -0.0050] | LOSS | 0.4734 | -0.0339 | -0.0509 |
| hotpotqa | in-domain | 7405 | 0.9065 | 0.8417 | -0.0648 [-0.0693, -0.0602] | LOSS | 0.6846 | -0.1269 | -0.0074 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.8624 | -0.0100 [-0.0132, -0.0067] | LOSS | 0.6079 | -0.0258 | +0.0103 |
| webqsp | zero-shot | 1503 | 0.3245 | 0.2819 | -0.0427 [-0.0590, -0.0258] | LOSS | 0.0535 | -0.0353 | -0.0486 |

## Against outputs\step1\fits\L-musique (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7834 | +0.1297 [+0.1245, +0.1349] | GAIN | 0.0047 | +0.1321 | +0.2381 |
| squad | in-domain | 11873 | 0.9100 | 0.9132 | +0.0033 [+0.0011, +0.0056] | WITHIN | 0.9054 | +0.0033 | +0.0006 |
| musique | zero-shot | 2417 | 0.2696 | 0.4978 | +0.2282 [+0.2165, +0.2402] | GAIN | 0.4734 | +0.1030 | +0.3045 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.8417 | -0.0598 [-0.0647, -0.0551] | LOSS | 0.6846 | -0.1188 | +0.0057 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8624 | -0.0108 [-0.0141, -0.0075] | LOSS | 0.6079 | -0.0236 | +0.0081 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.2819 | +0.1040 [+0.0806, +0.1274] | GAIN | 0.0535 | +0.0812 | +0.0659 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
