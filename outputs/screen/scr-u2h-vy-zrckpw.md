# Screen scr-u2h-zrckpw (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2g-y-zrckp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8009 | 0.8007 | -0.0002 [-0.0011, +0.0007] | WITHIN | 0.0047 | -0.0001 | +0.0012 |
| squad | in-domain | 11873 | 0.9132 | 0.9106 | -0.0026 [-0.0048, -0.0005] | WITHIN | 0.9054 | -0.0026 | +0.0002 |
| musique | zero-shot | 2417 | 0.5315 | 0.5131 | -0.0184 [-0.0249, -0.0123] | LOSS | 0.4734 | -0.0199 | +0.0079 |
| hotpotqa | in-domain | 7405 | 0.8512 | 0.9053 | +0.0540 [+0.0494, +0.0585] | GAIN | 0.6846 | +0.1083 | +0.0050 |
| 2wiki | in-domain | 12576 | 0.9171 | 0.9384 | +0.0213 [+0.0185, +0.0241] | GAIN | 0.6079 | +0.0553 | -0.0065 |
| webqsp | zero-shot | 1503 | 0.3523 | 0.3558 | +0.0035 [-0.0070, +0.0145] | WITHIN | 0.0535 | +0.0053 | -0.0007 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
