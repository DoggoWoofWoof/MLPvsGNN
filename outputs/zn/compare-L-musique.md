# Screen scr-zn (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrct (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7782 | 0.2212 | -0.5570 [-0.5659, -0.5484] | LOSS | 0.0047 | -0.5174 | -0.6926 |
| squad | in-domain | 11873 | 0.9105 | 0.9121 | +0.0016 [-0.0008, +0.0040] | WITHIN | 0.9054 | +0.0016 | -0.0005 |
| musique | zero-shot | 2417 | 0.5235 | 0.4783 | -0.0453 [-0.0548, -0.0358] | LOSS | 0.4734 | -0.0612 | -0.2147 |
| hotpotqa | in-domain | 7405 | 0.9041 | 0.8018 | -0.1022 [-0.1075, -0.0968] | LOSS | 0.6846 | -0.2026 | -0.0063 |
| 2wiki | in-domain | 12576 | 0.8733 | 0.7996 | -0.0737 [-0.0772, -0.0698] | LOSS | 0.6079 | -0.1500 | +0.0000 |
| webqsp | zero-shot | 1503 | 0.3384 | 0.0943 | -0.2441 [-0.2696, -0.2203] | LOSS | 0.0535 | -0.1929 | -0.0665 |

## Against outputs\screen\fits\scr-zspg (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7794 | 0.2212 | -0.5582 [-0.5670, -0.5495] | LOSS | 0.0047 | -0.5183 | -0.6957 |
| squad | in-domain | 11873 | 0.9125 | 0.9121 | -0.0004 [-0.0029, +0.0020] | WITHIN | 0.9054 | -0.0004 | +0.0025 |
| musique | zero-shot | 2417 | 0.5111 | 0.4783 | -0.0329 [-0.0433, -0.0226] | LOSS | 0.4734 | -0.0513 | -0.1953 |
| hotpotqa | in-domain | 7405 | 0.9065 | 0.8018 | -0.1047 [-0.1101, -0.0993] | LOSS | 0.6846 | -0.2054 | -0.0088 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.7996 | -0.0728 [-0.0766, -0.0690] | LOSS | 0.6079 | -0.1534 | +0.0100 |
| webqsp | zero-shot | 1503 | 0.3245 | 0.0943 | -0.2302 [-0.2552, -0.2061] | LOSS | 0.0535 | -0.1796 | -0.1284 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
