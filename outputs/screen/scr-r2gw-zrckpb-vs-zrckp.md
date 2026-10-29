# Screen scr-r2gw-zrckpb (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0 (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4393 | 0.4578 | +0.0185 [+0.0121, +0.0249] | GAIN | 0.0047 | +0.0013 | +0.0013 |
| squad | in-domain | 11873 | 0.9106 | 0.9080 | -0.0025 [-0.0046, -0.0004] | WITHIN | 0.9054 | -0.0025 | -0.0058 |
| musique | zero-shot | 2417 | 0.4972 | 0.4999 | +0.0027 [-0.0044, +0.0093] | WITHIN | 0.4734 | +0.0012 | +0.0397 |
| hotpotqa | in-domain | 7405 | 0.9051 | 0.8994 | -0.0057 [-0.0082, -0.0034] | WITHIN | 0.6846 | -0.0122 | -0.0003 |
| 2wiki | in-domain | 12576 | 0.9315 | 0.9274 | -0.0042 [-0.0058, -0.0025] | WITHIN | 0.6079 | -0.0130 | -0.0024 |
| webqsp | in-domain | 1503 | 0.6433 | 0.6577 | +0.0144 [+0.0034, +0.0257] | GAIN | 0.0535 | +0.0133 | +0.0027 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
