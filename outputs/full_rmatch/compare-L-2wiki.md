# Screen L-2wiki (p@swa, s1eval carves)

## Against outputs\step1\fits\L-2wiki (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6468 | 0.7803 | +0.1334 [+0.1281, +0.1386] | GAIN | 0.0047 | +0.1380 | +0.2663 |
| squad | in-domain | 11873 | 0.9112 | 0.9100 | -0.0013 [-0.0032, +0.0008] | WITHIN | 0.9054 | -0.0013 | -0.0024 |
| musique | in-domain | 2417 | 0.5601 | 0.5576 | -0.0025 [-0.0086, +0.0031] | WITHIN | 0.4734 | -0.0037 | -0.0033 |
| hotpotqa | in-domain | 7405 | 0.8977 | 0.9006 | +0.0029 [+0.0004, +0.0055] | WITHIN | 0.6846 | +0.0043 | +0.0070 |
| 2wiki | zero-shot | 12576 | 0.8074 | 0.7937 | -0.0136 [-0.0158, -0.0112] | LOSS | 0.6079 | -0.0236 | +0.0006 |
| webqsp | zero-shot | 1503 | 0.1075 | 0.2607 | +0.1532 [+0.1309, +0.1773] | GAIN | 0.0535 | +0.1257 | +0.0712 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
