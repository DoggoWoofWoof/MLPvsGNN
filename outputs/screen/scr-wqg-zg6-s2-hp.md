# Screen scr-wqg-zg6-s2-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zg6-s0-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4167 | 0.4192 | +0.0025 [-0.0030, +0.0080] | WITHIN | 0.0047 | -0.0029 | +0.0176 |
| squad | in-domain | 11873 | 0.9114 | 0.9117 | +0.0003 [-0.0016, +0.0023] | WITHIN | 0.9054 | +0.0003 | +0.0014 |
| musique | in-domain | 2417 | 0.5961 | 0.5944 | -0.0017 [-0.0067, +0.0036] | WITHIN | 0.4734 | +0.0050 | -0.0058 |
| hotpotqa | zero-shot | 7405 | 0.8183 | 0.8195 | +0.0012 [-0.0022, +0.0045] | WITHIN | 0.6846 | +0.0038 | -0.0327 |
| 2wiki | in-domain | 12576 | 0.9414 | 0.9421 | +0.0007 [-0.0008, +0.0022] | WITHIN | 0.6079 | +0.0020 | +0.0016 |
| webqsp | in-domain | 1503 | 0.6421 | 0.6311 | -0.0110 [-0.0223, +0.0006] | WITHIN | 0.0535 | -0.0080 | +0.0027 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
