# Screen scr-r2ws-zrckpb-s1-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s1-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4193 | 0.4281 | +0.0087 [+0.0024, +0.0149] | GAIN | 0.0047 | +0.0052 | +0.0412 |
| squad | in-domain | 11873 | 0.9115 | 0.9089 | -0.0026 [-0.0048, -0.0004] | WITHIN | 0.9054 | -0.0026 | -0.0035 |
| musique | in-domain | 2417 | 0.5768 | 0.5593 | -0.0175 [-0.0237, -0.0114] | LOSS | 0.4734 | -0.0112 | -0.0484 |
| hotpotqa | zero-shot | 7405 | 0.8547 | 0.8422 | -0.0125 [-0.0160, -0.0090] | LOSS | 0.6846 | -0.0234 | -0.0100 |
| 2wiki | in-domain | 12576 | 0.9284 | 0.9306 | +0.0022 [+0.0004, +0.0039] | WITHIN | 0.6079 | +0.0072 | -0.0103 |
| webqsp | in-domain | 1503 | 0.6417 | 0.6442 | +0.0024 [-0.0094, +0.0134] | WITHIN | 0.0535 | -0.0013 | -0.0113 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
