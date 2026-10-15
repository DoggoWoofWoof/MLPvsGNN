# Screen scr-g2-zg2-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3-zspk-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8040 | 0.8033 | -0.0007 [-0.0016, +0.0002] | WITHIN | 0.0047 | -0.0003 | -0.0034 |
| squad | in-domain | 11873 | 0.9135 | 0.9141 | +0.0006 [-0.0013, +0.0024] | WITHIN | 0.9054 | +0.0006 | -0.0007 |
| musique | in-domain | 2417 | 0.5754 | 0.5792 | +0.0038 [-0.0010, +0.0088] | WITHIN | 0.4734 | -0.0033 | -0.0004 |
| hotpotqa | zero-shot | 7405 | 0.8499 | 0.8499 | +0.0000 [-0.0027, +0.0027] | WITHIN | 0.6846 | -0.0001 | +0.0041 |
| 2wiki | in-domain | 12576 | 0.9333 | 0.9313 | -0.0020 [-0.0035, -0.0005] | WITHIN | 0.6079 | -0.0060 | +0.0032 |
| webqsp | zero-shot | 1503 | 0.2680 | 0.2890 | +0.0210 [+0.0110, +0.0309] | GAIN | 0.0535 | +0.0153 | -0.0013 |

## Against outputs\screen\fits\scr-g2-zg2t-hp (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8061 | 0.8033 | -0.0028 [-0.0037, -0.0019] | WITHIN | 0.0047 | -0.0026 | -0.0039 |
| squad | in-domain | 11873 | 0.9121 | 0.9141 | +0.0020 [+0.0002, +0.0040] | WITHIN | 0.9054 | +0.0020 | +0.0018 |
| musique | in-domain | 2417 | 0.5826 | 0.5792 | -0.0034 [-0.0088, +0.0018] | WITHIN | 0.4734 | -0.0058 | +0.0054 |
| hotpotqa | zero-shot | 7405 | 0.8431 | 0.8499 | +0.0068 [+0.0038, +0.0097] | WITHIN | 0.6846 | +0.0099 | +0.0113 |
| 2wiki | in-domain | 12576 | 0.9391 | 0.9313 | -0.0078 [-0.0094, -0.0063] | LOSS | 0.6079 | -0.0233 | +0.0003 |
| webqsp | zero-shot | 1503 | 0.2806 | 0.2890 | +0.0084 [-0.0000, +0.0171] | WITHIN | 0.0535 | +0.0060 | +0.0073 |

## Against outputs\screen\fits\scr-zspg-hp (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7807 | 0.8033 | +0.0226 [+0.0195, +0.0254] | GAIN | 0.0047 | +0.0194 | +0.0303 |
| squad | in-domain | 11873 | 0.9109 | 0.9141 | +0.0032 [+0.0011, +0.0054] | WITHIN | 0.9054 | +0.0032 | -0.0002 |
| musique | in-domain | 2417 | 0.5647 | 0.5792 | +0.0144 [+0.0067, +0.0222] | GAIN | 0.4734 | +0.0021 | +0.0434 |
| hotpotqa | zero-shot | 7405 | 0.8435 | 0.8499 | +0.0064 [+0.0016, +0.0111] | WITHIN | 0.6846 | +0.0112 | -0.0393 |
| 2wiki | in-domain | 12576 | 0.8744 | 0.9313 | +0.0569 [+0.0536, +0.0603] | GAIN | 0.6079 | +0.1063 | +0.0735 |
| webqsp | zero-shot | 1503 | 0.2262 | 0.2890 | +0.0628 [+0.0434, +0.0808] | GAIN | 0.0535 | +0.0446 | +0.0259 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8033 | +0.1493 [+0.1436, +0.1547] | GAIN | 0.0047 | +0.1535 | +0.2865 |
| squad | in-domain | 11873 | 0.9116 | 0.9141 | +0.0024 [+0.0003, +0.0048] | WITHIN | 0.9054 | +0.0024 | +0.0017 |
| musique | in-domain | 2417 | 0.5550 | 0.5792 | +0.0241 [+0.0161, +0.0320] | GAIN | 0.4734 | +0.0207 | +0.0496 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8499 | -0.0086 [-0.0136, -0.0037] | LOSS | 0.6846 | -0.0173 | -0.0355 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9313 | +0.0641 [+0.0607, +0.0676] | GAIN | 0.6079 | +0.1282 | +0.0767 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2890 | +0.1690 [+0.1448, +0.1930] | GAIN | 0.0535 | +0.1257 | +0.0898 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
