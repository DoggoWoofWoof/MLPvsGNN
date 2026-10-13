# P3a: the state-of-the-art rankers on our full corpora, beside our four models (s1eval)

Means over every s1eval question; R@5 / FC@5 / hit@1. The lean pair's webqsp read is zero-shot; the universal pair was fit on all six datasets' fit carves, webqsp's included. Records: outputs/p3a/.

## metaqa

| row | kind | R@5 | FC@5 | hit@1 |
| --- | --- | ---: | ---: | ---: |
| A1 | SOTA | 0.0096 | 0.0057 | **0.0073** |
| A2 | SOTA | 0.0058 | 0.0038 | 0.0031 |
| LINKED | SOTA | **0.2406** | **0.2075** | 0.0010 |
| DENSE | SOTA | 0.0087 | 0.0047 | 0.0019 |
| SPLADE | SOTA | 0.0030 | 0.0017 | 0.0010 |
| RRF | SOTA | 0.0047 | 0.0026 | 0.0008 |
| A1_all | SOTA | 0.0079 | 0.0043 | 0.0064 |
| A2_all | SOTA | 0.0054 | 0.0034 | 0.0022 |
| ORACLE | exposure | 0.3301 | 0.2802 | 0.0000 |
| A3 | our pool walk | 0.0058 | 0.0038 | 0.0029 |
| A0 | our pool walk | 0.0097 | 0.0057 | 0.0081 |
| zrc | model | 0.7796 | 0.6668 | 0.8920 |
| zsp | model | 0.7794 | 0.6672 | 0.8885 |
| twin0 | model | 0.7212 | 0.5960 | 0.7707 |
| gnn0 | model | 0.7839 | 0.6668 | 0.8889 |

| model | R@5 minus the best SOTA row [95%] | FC@5 minus the best SOTA row [95%] | hit@1 minus the best SOTA row [95%] | label | call |
| --- | --- | --- | --- | --- | --- |
| zrc | +0.5390 [+0.5300, +0.5471] vs LINKED | +0.4594 [+0.4492, +0.4688] vs LINKED | +0.8847 [+0.8780, +0.8911] vs A1 | ABOVE | AT_OR_ABOVE |
| zsp | +0.5388 [+0.5300, +0.5469] vs LINKED | +0.4598 [+0.4496, +0.4695] vs LINKED | +0.8812 [+0.8745, +0.8876] vs A1 | ABOVE | AT_OR_ABOVE |
| twin0 | +0.4806 [+0.4718, +0.4889] vs LINKED | +0.3886 [+0.3788, +0.3984] vs LINKED | +0.7634 [+0.7552, +0.7720] vs A1 | ABOVE | AT_OR_ABOVE |
| gnn0 | +0.5433 [+0.5347, +0.5517] vs LINKED | +0.4594 [+0.4495, +0.4690] vs LINKED | +0.8817 [+0.8750, +0.8880] vs A1 | ABOVE | AT_OR_ABOVE |

- The MLP's share of the GNN's gain over RRF: lean R@5 1.0, FC@5 0.999, hit@1 1.004; universal R@5 0.92, FC@5 0.893, hit@1 0.867.

## squad

| row | kind | R@5 | FC@5 | hit@1 |
| --- | --- | ---: | ---: | ---: |
| A1 | SOTA | **0.9053** | **0.9053** | **0.7349** |
| A2 | SOTA | 0.9053 | 0.9053 | 0.7349 |
| DENSE | SOTA | 0.8769 | 0.8769 | 0.6939 |
| SPLADE | SOTA | 0.8889 | 0.8889 | 0.7270 |
| RRF | SOTA | 0.9053 | 0.9053 | 0.7349 |
| A1_all | SOTA | 0.7993 | 0.7993 | 0.2074 |
| A2_all | SOTA | 0.7181 | 0.7181 | 0.1546 |
| A3 | our pool walk | 0.9053 | 0.9053 | 0.7349 |
| A0 | our pool walk | 0.9053 | 0.9053 | 0.7349 |
| zrc | model | 0.9107 | 0.9107 | 0.7505 |
| zsp | model | 0.9099 | 0.9099 | 0.7493 |
| twin0 | model | 0.8971 | 0.8971 | 0.7363 |
| gnn0 | model | 0.9038 | 0.9038 | 0.7431 |

| model | R@5 minus the best SOTA row [95%] | FC@5 minus the best SOTA row [95%] | hit@1 minus the best SOTA row [95%] | label | call |
| --- | --- | --- | --- | --- | --- |
| zrc | +0.0054 [+0.0023, +0.0086] vs A1 | +0.0054 [+0.0023, +0.0086] vs A1 | +0.0156 [+0.0109, +0.0200] vs A1 | ABOVE | AT_OR_ABOVE |
| zsp | +0.0045 [+0.0014, +0.0077] vs A1 | +0.0045 [+0.0014, +0.0077] vs A1 | +0.0143 [+0.0099, +0.0188] vs A1 | ABOVE | AT_OR_ABOVE |
| twin0 | -0.0083 [-0.0122, -0.0046] vs A1 | -0.0083 [-0.0122, -0.0046] vs A1 | +0.0013 [-0.0040, +0.0063] vs A1 | BELOW | BELOW |
| gnn0 | -0.0015 [-0.0052, +0.0019] vs A1 | -0.0015 [-0.0052, +0.0019] vs A1 | +0.0082 [+0.0034, +0.0131] vs A1 | ABOVE | AT_OR_ABOVE |

- The MLP's share of the GNN's gain over RRF: lean R@5 1.174, FC@5 1.174, hit@1 1.083; universal R@5 n/a, FC@5 n/a, hit@1 0.171.

## musique

| row | kind | R@5 | FC@5 | hit@1 |
| --- | --- | ---: | ---: | ---: |
| A1 | SOTA | 0.4253 | 0.1179 | 0.2764 |
| A2 | SOTA | 0.3795 | 0.0894 | 0.2213 |
| LINKED | SOTA | 0.2921 | 0.0943 | 0.3364 |
| DENSE | SOTA | 0.4572 | **0.1361** | 0.6016 |
| SPLADE | SOTA | 0.4314 | 0.1030 | 0.6264 |
| RRF | SOTA | **0.4734** | 0.1353 | **0.6347** |
| A1_all | SOTA | 0.4253 | 0.1179 | 0.2764 |
| A2_all | SOTA | 0.3795 | 0.0894 | 0.2213 |
| A3 | our pool walk | 0.3441 | 0.0724 | 0.2164 |
| A0 | our pool walk | 0.3776 | 0.1067 | 0.2280 |
| zrc | model | 0.5669 | 0.2586 | 0.7054 |
| zsp | model | 0.5657 | 0.2586 | 0.6951 |
| twin0 | model | 0.5119 | 0.1990 | 0.6500 |
| gnn0 | model | 0.5380 | 0.2491 | 0.6496 |

| model | R@5 minus the best SOTA row [95%] | FC@5 minus the best SOTA row [95%] | hit@1 minus the best SOTA row [95%] | label | call |
| --- | --- | --- | --- | --- | --- |
| zrc | +0.0934 [+0.0844, +0.1029] vs RRF | +0.1225 [+0.1067, +0.1374] vs DENSE | +0.0707 [+0.0554, +0.0856] vs RRF | ABOVE | AT_OR_ABOVE |
| zsp | +0.0923 [+0.0830, +0.1010] vs RRF | +0.1225 [+0.1063, +0.1374] vs DENSE | +0.0604 [+0.0447, +0.0761] vs RRF | ABOVE | AT_OR_ABOVE |
| twin0 | +0.0384 [+0.0289, +0.0486] vs RRF | +0.0629 [+0.0480, +0.0782] vs DENSE | +0.0153 [-0.0021, +0.0327] vs RRF | ABOVE | AT_OR_ABOVE |
| gnn0 | +0.0645 [+0.0543, +0.0749] vs RRF | +0.1129 [+0.0968, +0.1291] vs DENSE | +0.0149 [-0.0029, +0.0323] vs RRF | ABOVE | AT_OR_ABOVE |

- The MLP's share of the GNN's gain over RRF: lean R@5 1.013, FC@5 1.0, hit@1 1.171; universal R@5 0.596, FC@5 0.56, hit@1 1.027.

## hotpotqa

| row | kind | R@5 | FC@5 | hit@1 |
| --- | --- | ---: | ---: | ---: |
| A1 | SOTA | 0.6766 | **0.4366** | 0.4733 |
| A2 | SOTA | 0.4626 | 0.2146 | 0.1901 |
| LINKED | SOTA | 0.2341 | 0.1334 | 0.1161 |
| DENSE | SOTA | 0.6577 | 0.3854 | 0.8016 |
| SPLADE | SOTA | 0.6499 | 0.3687 | 0.8011 |
| RRF | SOTA | **0.6845** | 0.4240 | **0.8290** |
| A1_all | SOTA | 0.6570 | 0.4016 | 0.4966 |
| A2_all | SOTA | 0.5007 | 0.2423 | 0.2014 |
| A3 | our pool walk | 0.1640 | 0.0483 | 0.0944 |
| A0 | our pool walk | 0.6257 | 0.4073 | 0.4652 |
| zrc | model | 0.9049 | 0.8478 | 0.8515 |
| zsp | model | 0.9051 | 0.8488 | 0.8558 |
| twin0 | model | 0.8786 | 0.7947 | 0.8325 |
| gnn0 | model | 0.8993 | 0.8363 | 0.8664 |

| model | R@5 minus the best SOTA row [95%] | FC@5 minus the best SOTA row [95%] | hit@1 minus the best SOTA row [95%] | label | call |
| --- | --- | --- | --- | --- | --- |
| zrc | +0.2204 [+0.2143, +0.2265] vs RRF | +0.4112 [+0.3993, +0.4230] vs A1 | +0.0224 [+0.0151, +0.0298] vs RRF | ABOVE | AT_OR_ABOVE |
| zsp | +0.2205 [+0.2145, +0.2269] vs RRF | +0.4122 [+0.4007, +0.4242] vs A1 | +0.0267 [+0.0197, +0.0336] vs RRF | ABOVE | AT_OR_ABOVE |
| twin0 | +0.1941 [+0.1881, +0.2003] vs RRF | +0.3581 [+0.3468, +0.3698] vs A1 | +0.0035 [-0.0043, +0.0116] vs RRF | ABOVE | AT_OR_ABOVE |
| gnn0 | +0.2147 [+0.2086, +0.2211] vs RRF | +0.3997 [+0.3883, +0.4112] vs A1 | +0.0374 [+0.0305, +0.0443] vs RRF | ABOVE | AT_OR_ABOVE |

- The MLP's share of the GNN's gain over RRF: lean R@5 0.999, FC@5 0.998, hit@1 0.84; universal R@5 0.904, FC@5 0.899, hit@1 0.094.

## 2wiki

| row | kind | R@5 | FC@5 | hit@1 |
| --- | --- | ---: | ---: | ---: |
| A1 | SOTA | 0.5729 | 0.2435 | 0.3727 |
| A2 | SOTA | 0.4479 | 0.1327 | 0.1795 |
| LINKED | SOTA | 0.2297 | 0.0703 | 0.1207 |
| DENSE | SOTA | 0.5171 | 0.1621 | 0.7374 |
| SPLADE | SOTA | 0.6055 | **0.2556** | 0.8666 |
| RRF | SOTA | **0.6079** | 0.2537 | **0.8754** |
| A1_all | SOTA | 0.5590 | 0.2126 | 0.3773 |
| A2_all | SOTA | 0.4439 | 0.1412 | 0.1837 |
| A3 | our pool walk | 0.1541 | 0.0308 | 0.0460 |
| A0 | our pool walk | 0.4479 | 0.1656 | 0.2864 |
| zrc | model | 0.8697 | 0.7055 | 0.8936 |
| zsp | model | 0.8726 | 0.7205 | 0.8983 |
| twin0 | model | 0.8487 | 0.6660 | 0.8709 |
| gnn0 | model | 0.8906 | 0.7588 | 0.9191 |

| model | R@5 minus the best SOTA row [95%] | FC@5 minus the best SOTA row [95%] | hit@1 minus the best SOTA row [95%] | label | call |
| --- | --- | --- | --- | --- | --- |
| zrc | +0.2618 [+0.2570, +0.2663] vs RRF | +0.4498 [+0.4405, +0.4588] vs SPLADE | +0.0182 [+0.0125, +0.0237] vs RRF | ABOVE | AT_OR_ABOVE |
| zsp | +0.2646 [+0.2598, +0.2693] vs RRF | +0.4649 [+0.4555, +0.4741] vs SPLADE | +0.0229 [+0.0174, +0.0284] vs RRF | ABOVE | AT_OR_ABOVE |
| twin0 | +0.2407 [+0.2361, +0.2453] vs RRF | +0.4103 [+0.4008, +0.4189] vs SPLADE | -0.0045 [-0.0107, +0.0017] vs RRF | ABOVE | AT_OR_ABOVE |
| gnn0 | +0.2827 [+0.2779, +0.2874] vs RRF | +0.5032 [+0.4940, +0.5122] vs SPLADE | +0.0437 [+0.0384, +0.0489] vs RRF | ABOVE | AT_OR_ABOVE |

- The MLP's share of the GNN's gain over RRF: lean R@5 0.989, FC@5 0.968, hit@1 0.795; universal R@5 0.852, FC@5 0.816, hit@1 -0.103.

## webqsp

| row | kind | R@5 | FC@5 | hit@1 |
| --- | --- | ---: | ---: | ---: |
| A1 | SOTA | 0.1006 | 0.0798 | 0.0645 |
| A2 | SOTA | 0.0462 | 0.0353 | 0.0273 |
| LINKED | SOTA | 0.0114 | 0.0080 | 0.0067 |
| DENSE | SOTA | **0.1095** | **0.0878** | **0.0712** |
| SPLADE | SOTA | 0.0367 | 0.0246 | 0.0386 |
| RRF | SOTA | 0.0535 | 0.0379 | 0.0452 |
| A1_all | SOTA | 0.1006 | 0.0798 | 0.0645 |
| A2_all | SOTA | 0.0462 | 0.0353 | 0.0273 |
| ORACLE | exposure | 0.0222 | 0.0160 | 0.0073 |
| A3 | our pool walk | 0.0356 | 0.0266 | 0.0286 |
| A0 | our pool walk | 0.0978 | 0.0778 | 0.0725 |
| zrc | model | 0.3071 | 0.2415 | 0.1104 |
| zsp | model | 0.2677 | 0.2089 | 0.1344 |
| twin0 | model | 0.6015 | 0.4877 | 0.5629 |
| gnn0 | model | 0.6289 | 0.5110 | 0.6367 |

| model | R@5 minus the best SOTA row [95%] | FC@5 minus the best SOTA row [95%] | hit@1 minus the best SOTA row [95%] | label | call |
| --- | --- | --- | --- | --- | --- |
| zrc | +0.1976 [+0.1732, +0.2221] vs DENSE | +0.1537 [+0.1297, +0.1776] vs DENSE | +0.0393 [+0.0180, +0.0605] vs DENSE | ABOVE | AT_OR_ABOVE |
| zsp | +0.1582 [+0.1357, +0.1824] vs DENSE | +0.1211 [+0.0991, +0.1444] vs DENSE | +0.0632 [+0.0406, +0.0872] vs DENSE | ABOVE | AT_OR_ABOVE |
| twin0 | +0.4920 [+0.4690, +0.5148] vs DENSE | +0.3999 [+0.3746, +0.4238] vs DENSE | +0.4917 [+0.4651, +0.5190] vs DENSE | ABOVE | AT_OR_ABOVE |
| gnn0 | +0.5194 [+0.4965, +0.5416] vs DENSE | +0.4232 [+0.3965, +0.4471] vs DENSE | +0.5655 [+0.5403, +0.5915] vs DENSE | ABOVE | AT_OR_ABOVE |

- The MLP's share of the GNN's gain over RRF: lean R@5 1.184, FC@5 1.191, hit@1 0.731; universal R@5 0.952, FC@5 0.951, hit@1 0.875.

## Claim 2 on our corpora

- zsp: **SOTA_GNN**.
- gnn0: **SOTA_GNN**.
