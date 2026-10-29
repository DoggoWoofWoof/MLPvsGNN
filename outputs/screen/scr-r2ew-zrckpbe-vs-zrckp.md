# Screen scr-r2ew-zrckpbe (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0 (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4393 | 0.4488 | +0.0095 [+0.0037, +0.0154] | GAIN | 0.0047 | -0.0028 | -0.0068 |
| squad | in-domain | 11873 | 0.9106 | 0.9080 | -0.0025 [-0.0046, -0.0006] | WITHIN | 0.9054 | -0.0025 | -0.0057 |
| musique | zero-shot | 2417 | 0.4972 | 0.4911 | -0.0061 [-0.0129, +0.0005] | WITHIN | 0.4734 | -0.0025 | +0.0248 |
| hotpotqa | in-domain | 7405 | 0.9051 | 0.9034 | -0.0018 [-0.0042, +0.0007] | WITHIN | 0.6846 | -0.0028 | -0.0053 |
| 2wiki | in-domain | 12576 | 0.9315 | 0.9322 | +0.0007 [-0.0009, +0.0023] | WITHIN | 0.6079 | +0.0002 | -0.0134 |
| webqsp | in-domain | 1503 | 0.6433 | 0.6506 | +0.0073 [-0.0037, +0.0181] | WITHIN | 0.0535 | +0.0086 | -0.0027 |

## Against outputs\screen\fits\scr-r2gw-zrckpb (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4578 | 0.4488 | -0.0090 [-0.0150, -0.0026] | LOSS | 0.0047 | -0.0041 | -0.0082 |
| squad | in-domain | 11873 | 0.9080 | 0.9080 | +0.0000 [-0.0022, +0.0022] | WITHIN | 0.9054 | +0.0000 | +0.0001 |
| musique | zero-shot | 2417 | 0.4999 | 0.4911 | -0.0088 [-0.0153, -0.0018] | LOSS | 0.4734 | -0.0037 | -0.0149 |
| hotpotqa | in-domain | 7405 | 0.8994 | 0.9034 | +0.0040 [+0.0015, +0.0066] | WITHIN | 0.6846 | +0.0093 | -0.0050 |
| 2wiki | in-domain | 12576 | 0.9274 | 0.9322 | +0.0049 [+0.0032, +0.0065] | WITHIN | 0.6079 | +0.0133 | -0.0110 |
| webqsp | in-domain | 1503 | 0.6577 | 0.6506 | -0.0071 [-0.0175, +0.0024] | WITHIN | 0.0535 | -0.0047 | -0.0053 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
