# Screen scr-p1c3-zspk-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zspg-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7807 | 0.8040 | +0.0233 [+0.0204, +0.0261] | GAIN | 0.0047 | +0.0197 | +0.0336 |
| squad | in-domain | 11873 | 0.9109 | 0.9135 | +0.0026 [+0.0004, +0.0048] | WITHIN | 0.9054 | +0.0026 | +0.0005 |
| musique | in-domain | 2417 | 0.5647 | 0.5754 | +0.0107 [+0.0030, +0.0182] | GAIN | 0.4734 | +0.0054 | +0.0439 |
| hotpotqa | zero-shot | 7405 | 0.8435 | 0.8499 | +0.0064 [+0.0017, +0.0110] | WITHIN | 0.6846 | +0.0113 | -0.0433 |
| 2wiki | in-domain | 12576 | 0.8744 | 0.9333 | +0.0589 [+0.0555, +0.0623] | GAIN | 0.6079 | +0.1123 | +0.0703 |
| webqsp | zero-shot | 1503 | 0.2262 | 0.2680 | +0.0418 [+0.0228, +0.0595] | GAIN | 0.0535 | +0.0293 | +0.0273 |

## Against outputs\screen\fits\scr-p1c2-zsp-hp (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8008 | 0.8040 | +0.0032 [+0.0019, +0.0045] | WITHIN | 0.0047 | +0.0022 | +0.0071 |
| squad | in-domain | 11873 | 0.9111 | 0.9135 | +0.0024 [+0.0003, +0.0047] | WITHIN | 0.9054 | +0.0024 | -0.0003 |
| musique | in-domain | 2417 | 0.5707 | 0.5754 | +0.0047 [-0.0029, +0.0118] | WITHIN | 0.4734 | -0.0017 | +0.0306 |
| hotpotqa | zero-shot | 7405 | 0.8475 | 0.8499 | +0.0024 [-0.0016, +0.0064] | WITHIN | 0.6846 | +0.0012 | -0.0363 |
| 2wiki | in-domain | 12576 | 0.9171 | 0.9333 | +0.0162 [+0.0138, +0.0185] | GAIN | 0.6079 | +0.0311 | +0.0371 |
| webqsp | zero-shot | 1503 | 0.2603 | 0.2680 | +0.0077 [-0.0051, +0.0209] | WITHIN | 0.0535 | +0.0027 | +0.0093 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8040 | +0.1500 [+0.1444, +0.1554] | GAIN | 0.0047 | +0.1538 | +0.2898 |
| squad | in-domain | 11873 | 0.9116 | 0.9135 | +0.0019 [-0.0003, +0.0040] | WITHIN | 0.9054 | +0.0019 | +0.0024 |
| musique | in-domain | 2417 | 0.5550 | 0.5754 | +0.0204 [+0.0128, +0.0281] | GAIN | 0.4734 | +0.0240 | +0.0501 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8499 | -0.0086 [-0.0133, -0.0038] | LOSS | 0.6846 | -0.0172 | -0.0396 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9333 | +0.0661 [+0.0628, +0.0697] | GAIN | 0.6079 | +0.1341 | +0.0736 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2680 | +0.1480 [+0.1237, +0.1724] | GAIN | 0.0535 | +0.1104 | +0.0912 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
