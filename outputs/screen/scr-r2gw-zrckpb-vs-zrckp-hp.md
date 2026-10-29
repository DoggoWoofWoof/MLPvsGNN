# Screen scr-r2gw-zrckpb-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3927 | 0.4459 | +0.0533 [+0.0475, +0.0590] | GAIN | 0.0047 | +0.0391 | +0.0424 |
| squad | in-domain | 11873 | 0.9117 | 0.9090 | -0.0028 [-0.0049, -0.0008] | WITHIN | 0.9054 | -0.0028 | -0.0050 |
| musique | in-domain | 2417 | 0.5743 | 0.5689 | -0.0055 [-0.0119, +0.0011] | WITHIN | 0.4734 | -0.0021 | -0.0228 |
| hotpotqa | zero-shot | 7405 | 0.8525 | 0.8498 | -0.0026 [-0.0059, +0.0008] | WITHIN | 0.6846 | -0.0045 | -0.0239 |
| 2wiki | in-domain | 12576 | 0.9299 | 0.9304 | +0.0005 [-0.0011, +0.0022] | WITHIN | 0.6079 | +0.0015 | -0.0127 |
| webqsp | in-domain | 1503 | 0.6382 | 0.6488 | +0.0106 [+0.0002, +0.0218] | GAIN | 0.0535 | +0.0053 | -0.0093 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
