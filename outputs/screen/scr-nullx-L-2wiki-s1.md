# Screen scr-nullx-L-2wiki-s1 (p@swa, s1eval carves)

## Against outputs\step1\fits\L-2wiki (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6468 | 0.6502 | +0.0033 [+0.0005, +0.0063] | WITHIN | 0.0047 | +0.0027 | +0.0027 |
| squad | in-domain | 11873 | 0.9112 | 0.9059 | -0.0053 [-0.0075, -0.0031] | WITHIN | 0.9054 | -0.0053 | -0.0003 |
| musique | in-domain | 2417 | 0.5601 | 0.5603 | +0.0002 [-0.0062, +0.0069] | WITHIN | 0.4734 | +0.0066 | -0.0062 |
| hotpotqa | in-domain | 7405 | 0.8977 | 0.9040 | +0.0063 [+0.0036, +0.0090] | WITHIN | 0.6846 | +0.0131 | +0.0028 |
| 2wiki | zero-shot | 12576 | 0.8074 | 0.8044 | -0.0030 [-0.0053, -0.0005] | WITHIN | 0.6079 | +0.0048 | -0.0123 |
| webqsp | zero-shot | 1503 | 0.1075 | 0.1318 | +0.0243 [+0.0145, +0.0349] | GAIN | 0.0535 | +0.0206 | +0.0007 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
