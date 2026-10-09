# Screen scr-zgr-hp-off (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrm-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7801 | 0.7569 | -0.0231 [-0.0254, -0.0211] | LOSS | 0.0047 | -0.0219 | -0.0657 |
| squad | in-domain | 11873 | 0.9116 | 0.9128 | +0.0013 [-0.0008, +0.0033] | WITHIN | 0.9054 | +0.0013 | -0.0002 |
| musique | in-domain | 2417 | 0.5600 | 0.5644 | +0.0044 [-0.0020, +0.0105] | WITHIN | 0.4734 | -0.0004 | -0.0062 |
| hotpotqa | zero-shot | 7405 | 0.8457 | 0.8467 | +0.0010 [-0.0024, +0.0041] | WITHIN | 0.6846 | +0.0003 | +0.0016 |
| 2wiki | in-domain | 12576 | 0.8681 | 0.8601 | -0.0079 [-0.0101, -0.0056] | LOSS | 0.6079 | -0.0212 | +0.0095 |
| webqsp | zero-shot | 1503 | 0.2416 | 0.2330 | -0.0086 [-0.0188, +0.0019] | WITHIN | 0.0535 | -0.0073 | -0.0166 |

## Against outputs\screen\fits\scr-zgr-hp (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7826 | 0.7569 | -0.0257 [-0.0279, -0.0234] | LOSS | 0.0047 | -0.0233 | -0.0724 |
| squad | in-domain | 11873 | 0.8975 | 0.9128 | +0.0153 [+0.0124, +0.0183] | GAIN | 0.9054 | +0.0153 | +0.0295 |
| musique | in-domain | 2417 | 0.5125 | 0.5644 | +0.0518 [+0.0439, +0.0600] | GAIN | 0.4734 | +0.0480 | +0.1171 |
| hotpotqa | zero-shot | 7405 | 0.8194 | 0.8467 | +0.0273 [+0.0230, +0.0313] | GAIN | 0.6846 | +0.0496 | +0.0462 |
| 2wiki | in-domain | 12576 | 0.8656 | 0.8601 | -0.0054 [-0.0078, -0.0031] | WITHIN | 0.6079 | -0.0183 | +0.0401 |
| webqsp | zero-shot | 1503 | 0.2350 | 0.2330 | -0.0021 [-0.0146, +0.0103] | WITHIN | 0.0535 | -0.0020 | -0.0120 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
