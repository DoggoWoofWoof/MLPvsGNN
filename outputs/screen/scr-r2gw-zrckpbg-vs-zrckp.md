# Screen scr-r2gw-zrckpbg (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0 (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4393 | 0.4697 | +0.0303 [+0.0239, +0.0365] | GAIN | 0.0047 | +0.0130 | -0.0002 |
| squad | in-domain | 11873 | 0.9106 | 0.9071 | -0.0035 [-0.0056, -0.0013] | WITHIN | 0.9054 | -0.0035 | -0.0043 |
| musique | zero-shot | 2417 | 0.4972 | 0.4990 | +0.0018 [-0.0057, +0.0086] | WITHIN | 0.4734 | +0.0029 | +0.0108 |
| hotpotqa | in-domain | 7405 | 0.9051 | 0.9002 | -0.0049 [-0.0073, -0.0025] | WITHIN | 0.6846 | -0.0090 | -0.0192 |
| 2wiki | in-domain | 12576 | 0.9315 | 0.9308 | -0.0007 [-0.0022, +0.0009] | WITHIN | 0.6079 | -0.0030 | -0.0091 |
| webqsp | in-domain | 1503 | 0.6433 | 0.6545 | +0.0112 [-0.0002, +0.0221] | WITHIN | 0.0535 | +0.0060 | -0.0007 |

## Against outputs\screen\fits\scr-r2gw-zrckpb (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4578 | 0.4697 | +0.0118 [+0.0069, +0.0162] | GAIN | 0.0047 | +0.0117 | -0.0015 |
| squad | in-domain | 11873 | 0.9080 | 0.9071 | -0.0009 [-0.0027, +0.0008] | WITHIN | 0.9054 | -0.0009 | +0.0015 |
| musique | zero-shot | 2417 | 0.4999 | 0.4990 | -0.0009 [-0.0059, +0.0043] | WITHIN | 0.4734 | +0.0017 | -0.0290 |
| hotpotqa | in-domain | 7405 | 0.8994 | 0.9002 | +0.0008 [-0.0016, +0.0030] | WITHIN | 0.6846 | +0.0031 | -0.0189 |
| 2wiki | in-domain | 12576 | 0.9274 | 0.9308 | +0.0035 [+0.0021, +0.0049] | WITHIN | 0.6079 | +0.0100 | -0.0068 |
| webqsp | in-domain | 1503 | 0.6577 | 0.6545 | -0.0032 [-0.0096, +0.0031] | WITHIN | 0.0535 | -0.0073 | -0.0033 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
