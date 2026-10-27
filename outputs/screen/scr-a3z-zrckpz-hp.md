# Screen scr-a3z-zrckpz-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3p-zrckp-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8044 | 0.8030 | -0.0014 [-0.0023, -0.0006] | WITHIN | 0.0047 | -0.0020 | +0.0005 |
| squad | in-domain | 11873 | 0.9127 | 0.9124 | -0.0003 [-0.0020, +0.0017] | WITHIN | 0.9054 | -0.0003 | +0.0004 |
| musique | in-domain | 2417 | 0.5745 | 0.5722 | -0.0023 [-0.0077, +0.0029] | WITHIN | 0.4734 | -0.0008 | -0.0033 |
| hotpotqa | zero-shot | 7405 | 0.8513 | 0.8542 | +0.0029 [-0.0002, +0.0060] | WITHIN | 0.6846 | +0.0057 | -0.0153 |
| 2wiki | in-domain | 12576 | 0.9350 | 0.9346 | -0.0005 [-0.0021, +0.0012] | WITHIN | 0.6079 | -0.0015 | -0.0019 |
| webqsp | zero-shot | 1503 | 0.3475 | 0.3514 | +0.0039 [-0.0067, +0.0144] | WITHIN | 0.0535 | +0.0033 | +0.0033 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8030 | +0.1490 [+0.1434, +0.1544] | GAIN | 0.0047 | +0.1519 | +0.2889 |
| squad | in-domain | 11873 | 0.9116 | 0.9124 | +0.0008 [-0.0014, +0.0030] | WITHIN | 0.9054 | +0.0008 | +0.0029 |
| musique | in-domain | 2417 | 0.5550 | 0.5722 | +0.0172 [+0.0094, +0.0247] | GAIN | 0.4734 | +0.0174 | +0.0298 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8542 | -0.0043 [-0.0090, +0.0003] | WITHIN | 0.6846 | -0.0095 | -0.0497 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9346 | +0.0673 [+0.0641, +0.0709] | GAIN | 0.6079 | +0.1395 | +0.0627 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3514 | +0.2314 [+0.2058, +0.2556] | GAIN | 0.0535 | +0.1883 | +0.0719 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
