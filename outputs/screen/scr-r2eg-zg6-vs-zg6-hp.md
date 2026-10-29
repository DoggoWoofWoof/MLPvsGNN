# Screen scr-r2eg-zg6-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zg6-s0-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4167 | 0.3847 | -0.0320 [-0.0374, -0.0266] | LOSS | 0.0047 | -0.0312 | +0.0167 |
| squad | in-domain | 11873 | 0.9114 | 0.9108 | -0.0006 [-0.0029, +0.0018] | WITHIN | 0.9054 | -0.0006 | -0.0009 |
| musique | in-domain | 2417 | 0.5961 | 0.5927 | -0.0033 [-0.0092, +0.0024] | WITHIN | 0.4734 | -0.0012 | -0.0070 |
| hotpotqa | zero-shot | 7405 | 0.8183 | 0.8280 | +0.0097 [+0.0059, +0.0134] | GAIN | 0.6846 | +0.0185 | -0.0126 |
| 2wiki | in-domain | 12576 | 0.9414 | 0.9399 | -0.0015 [-0.0032, +0.0002] | WITHIN | 0.6079 | -0.0066 | -0.0025 |
| webqsp | in-domain | 1503 | 0.6421 | 0.6343 | -0.0078 [-0.0203, +0.0043] | WITHIN | 0.0535 | -0.0007 | +0.0027 |

## Against outputs\screen\fits\scr-wqk-zspk-s0-hp (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3384 | 0.3847 | +0.0463 [+0.0388, +0.0539] | GAIN | 0.0047 | +0.0418 | -0.0340 |
| squad | in-domain | 11873 | 0.9114 | 0.9108 | -0.0006 [-0.0030, +0.0019] | WITHIN | 0.9054 | -0.0006 | +0.0018 |
| musique | in-domain | 2417 | 0.5794 | 0.5927 | +0.0134 [+0.0067, +0.0205] | GAIN | 0.4734 | +0.0236 | +0.0306 |
| hotpotqa | zero-shot | 7405 | 0.8496 | 0.8280 | -0.0216 [-0.0258, -0.0171] | LOSS | 0.6846 | -0.0253 | -0.0352 |
| 2wiki | in-domain | 12576 | 0.9330 | 0.9399 | +0.0069 [+0.0050, +0.0088] | WITHIN | 0.6079 | +0.0190 | -0.0010 |
| webqsp | in-domain | 1503 | 0.6153 | 0.6343 | +0.0190 [+0.0046, +0.0335] | GAIN | 0.0535 | +0.0293 | +0.0106 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
