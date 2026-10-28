# Screen scr-psmk-zrckpg (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0 (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4393 | 0.4686 | +0.0293 [+0.0236, +0.0350] | GAIN | 0.0047 | +0.0199 | -0.0144 |
| squad | in-domain | 11873 | 0.9106 | 0.9079 | -0.0027 [-0.0049, -0.0007] | WITHIN | 0.9054 | -0.0027 | -0.0041 |
| musique | zero-shot | 2417 | 0.4972 | 0.5048 | +0.0076 [+0.0006, +0.0142] | GAIN | 0.4734 | +0.0000 | +0.0856 |
| hotpotqa | in-domain | 7405 | 0.9051 | 0.8797 | -0.0254 [-0.0286, -0.0221] | LOSS | 0.6846 | -0.0459 | -0.0404 |
| 2wiki | in-domain | 12576 | 0.9315 | 0.9200 | -0.0115 [-0.0133, -0.0097] | LOSS | 0.6079 | -0.0328 | +0.0150 |
| webqsp | in-domain | 1503 | 0.6433 | 0.6300 | -0.0133 [-0.0225, -0.0036] | LOSS | 0.0535 | -0.0153 | -0.0439 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
