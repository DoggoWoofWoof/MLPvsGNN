# Screen L-squad (p@swa, s1eval carves)

## Against outputs\full_zrct\fits\L-squad (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7799 | 0.7787 | -0.0013 [-0.0024, -0.0002] | WITHIN | 0.0047 | -0.0011 | -0.0063 |
| squad | zero-shot | 11873 | 0.8912 | 0.8837 | -0.0075 [-0.0101, -0.0049] | WITHIN | 0.9054 | -0.0075 | -0.0094 |
| musique | in-domain | 2417 | 0.5710 | 0.5699 | -0.0011 [-0.0069, +0.0050] | WITHIN | 0.4734 | -0.0046 | -0.0029 |
| hotpotqa | in-domain | 7405 | 0.9041 | 0.9059 | +0.0018 [-0.0006, +0.0042] | WITHIN | 0.6846 | +0.0038 | -0.0112 |
| 2wiki | in-domain | 12576 | 0.8687 | 0.8706 | +0.0019 [-0.0001, +0.0038] | WITHIN | 0.6079 | +0.0018 | -0.0002 |
| webqsp | zero-shot | 1503 | 0.2912 | 0.2966 | +0.0054 [-0.0131, +0.0222] | WITHIN | 0.0535 | +0.0053 | +0.0679 |

## Against outputs\full_zrm\fits\L-squad (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7792 | 0.7787 | -0.0006 [-0.0018, +0.0005] | WITHIN | 0.0047 | +0.0002 | -0.0071 |
| squad | zero-shot | 11873 | 0.8903 | 0.8837 | -0.0067 [-0.0093, -0.0042] | WITHIN | 0.9054 | -0.0067 | -0.0082 |
| musique | in-domain | 2417 | 0.5690 | 0.5699 | +0.0009 [-0.0049, +0.0070] | WITHIN | 0.4734 | +0.0025 | +0.0008 |
| hotpotqa | in-domain | 7405 | 0.9042 | 0.9059 | +0.0018 [-0.0008, +0.0042] | WITHIN | 0.6846 | +0.0022 | -0.0119 |
| 2wiki | in-domain | 12576 | 0.8689 | 0.8706 | +0.0017 [-0.0001, +0.0037] | WITHIN | 0.6079 | -0.0007 | -0.0049 |
| webqsp | zero-shot | 1503 | 0.2505 | 0.2966 | +0.0461 [+0.0322, +0.0609] | GAIN | 0.0535 | +0.0393 | +0.0506 |

## Against outputs\step1\fits\L-squad (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6535 | 0.7787 | +0.1252 [+0.1199, +0.1305] | GAIN | 0.0047 | +0.1269 | +0.2459 |
| squad | zero-shot | 11873 | 0.8797 | 0.8837 | +0.0040 [+0.0009, +0.0071] | WITHIN | 0.9054 | +0.0040 | +0.0148 |
| musique | in-domain | 2417 | 0.5743 | 0.5699 | -0.0044 [-0.0107, +0.0018] | WITHIN | 0.4734 | -0.0046 | -0.0054 |
| hotpotqa | in-domain | 7405 | 0.9045 | 0.9059 | +0.0015 [-0.0011, +0.0039] | WITHIN | 0.6846 | +0.0039 | +0.0092 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.8706 | -0.0017 [-0.0041, +0.0003] | WITHIN | 0.6079 | -0.0045 | +0.0031 |
| webqsp | zero-shot | 1503 | 0.1302 | 0.2966 | +0.1664 [+0.1435, +0.1898] | GAIN | 0.0535 | +0.1331 | +0.1164 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
