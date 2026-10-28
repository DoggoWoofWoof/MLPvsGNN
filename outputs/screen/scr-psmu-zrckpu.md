# Screen scr-psmu-zrckpu (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0 (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4393 | 0.4172 | -0.0221 [-0.0270, -0.0171] | LOSS | 0.0047 | -0.0188 | -0.0165 |
| squad | in-domain | 11873 | 0.9106 | 0.9100 | -0.0005 [-0.0023, +0.0013] | WITHIN | 0.9054 | -0.0005 | -0.0011 |
| musique | zero-shot | 2417 | 0.4972 | 0.4829 | -0.0143 [-0.0200, -0.0083] | LOSS | 0.4734 | -0.0099 | -0.0004 |
| hotpotqa | in-domain | 7405 | 0.9051 | 0.9030 | -0.0022 [-0.0045, +0.0001] | WITHIN | 0.6846 | -0.0042 | +0.0046 |
| 2wiki | in-domain | 12576 | 0.9315 | 0.9287 | -0.0028 [-0.0044, -0.0012] | WITHIN | 0.6079 | -0.0097 | +0.0031 |
| webqsp | in-domain | 1503 | 0.6433 | 0.6391 | -0.0042 [-0.0134, +0.0047] | WITHIN | 0.0535 | -0.0073 | -0.0166 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
