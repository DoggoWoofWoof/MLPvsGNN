# GNC: the cross-fitted first pass, its lists and its columns (decides nothing)

From docs/GNC_CROSSFIT_FIRST_PASS_GNN.md. R@5 of the assembled anchors (out of sample on every carve), GNR's in-sample R@5 beside it on the training carves, the base's s1eval R@5 on the s1eval carves; the T lists and the share of golds outside the anchors next to one, beside LTM's (rrf's anchors); the share of rows next to an anchor and the gold share among them, beside LTM's columns; whether an s1eval carve's columns equal GNR's.

| seed | split | carve | GNC first R@5 | GNR in sample | base R@5 | rrf R@5 | T lists (LTM) | golds next to an anchor (LTM) | rows next to an anchor (LTM) | gold share there (LTM) | = GNR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | WQ-musique | webqsp/fit | 0.5660 | 0.8190 | - | 0.0569 | 5982 (5995) | 0.406 (0.525) | 0.115 (0.050) | 0.003 (0.010) | - |
| 0 | WQ-musique | squad/fit | 0.9191 | 0.9278 | - | 0.9110 | 26596 (27791) | 0.625 (0.634) | 0.465 (0.473) | 0.002 (0.002) | - |
| 0 | WQ-musique | hotpotqa/fit | 0.8612 | 0.8802 | - | 0.6951 | 29464 (29362) | 0.693 (0.766) | 0.171 (0.115) | 0.001 (0.005) | - |
| 0 | WQ-musique | 2wiki/fit | 0.9387 | 0.9470 | - | 0.5981 | 29476 (29167) | 0.754 (0.730) | 0.125 (0.089) | 0.001 (0.008) | - |
| 0 | WQ-musique | metaqa/s1eval | 0.3615 | - | 0.3615 | 0.0047 | 48501 (48358) | 0.296 (0.115) | 0.092 (0.008) | 0.005 (0.025) | True |
| 0 | WQ-musique | squad/s1eval | 0.9142 | - | 0.9142 | 0.9054 | 54306 (56120) | 0.601 (0.586) | 0.417 (0.430) | 0.002 (0.002) | True |
| 0 | WQ-musique | musique/s1eval | 0.5086 | - | 0.5086 | 0.4734 | 12084 (12084) | 0.568 (0.511) | 0.128 (0.047) | 0.001 (0.003) | True |
| 0 | WQ-musique | hotpotqa/s1eval | 0.8593 | - | 0.8593 | 0.6846 | 36932 (36668) | 0.667 (0.755) | 0.169 (0.114) | 0.001 (0.006) | True |
| 0 | WQ-musique | 2wiki/s1eval | 0.9269 | - | 0.9269 | 0.6079 | 62841 (61921) | 0.806 (0.775) | 0.130 (0.093) | 0.001 (0.008) | True |
| 0 | WQ-musique | webqsp/s1eval | 0.6632 | - | 0.6632 | 0.0535 | 7502 (7500) | 0.349 (0.442) | 0.132 (0.048) | 0.002 (0.011) | True |
| 0 | WQ-hotpotqa | webqsp/fit | 0.5666 | 0.8265 | - | 0.0569 | 5982 (5995) | 0.400 (0.525) | 0.116 (0.050) | 0.003 (0.010) | - |
| 0 | WQ-hotpotqa | squad/fit | 0.9179 | 0.9310 | - | 0.9110 | 26863 (27791) | 0.603 (0.634) | 0.454 (0.473) | 0.002 (0.002) | - |
| 0 | WQ-hotpotqa | musique/s1fit | 0.6123 | 0.6818 | - | 0.5114 | 23001 (23002) | 0.528 (0.469) | 0.054 (0.045) | 0.002 (0.002) | - |
| 0 | WQ-hotpotqa | 2wiki/fit | 0.9381 | 0.9477 | - | 0.5981 | 29535 (29167) | 0.753 (0.730) | 0.119 (0.089) | 0.001 (0.008) | - |
| 0 | WQ-hotpotqa | metaqa/s1eval | 0.3519 | - | 0.3519 | 0.0047 | 48481 (48358) | 0.272 (0.115) | 0.066 (0.008) | 0.006 (0.025) | True |
| 0 | WQ-hotpotqa | squad/s1eval | 0.9139 | - | 0.9139 | 0.9054 | 54429 (56120) | 0.596 (0.586) | 0.408 (0.430) | 0.002 (0.002) | True |
| 0 | WQ-hotpotqa | musique/s1eval | 0.5912 | - | 0.5912 | 0.4734 | 12082 (12084) | 0.558 (0.511) | 0.068 (0.047) | 0.002 (0.003) | True |
| 0 | WQ-hotpotqa | hotpotqa/s1eval | 0.8004 | - | 0.8004 | 0.6846 | 36803 (36668) | 0.702 (0.755) | 0.163 (0.114) | 0.002 (0.006) | True |
| 0 | WQ-hotpotqa | 2wiki/s1eval | 0.9296 | - | 0.9296 | 0.6079 | 62738 (61921) | 0.784 (0.775) | 0.123 (0.093) | 0.001 (0.008) | True |
| 0 | WQ-hotpotqa | webqsp/s1eval | 0.6658 | - | 0.6658 | 0.0535 | 7498 (7500) | 0.321 (0.442) | 0.131 (0.048) | 0.002 (0.011) | True |
