# MHB: the multi-hop bridges our links miss (docs/MHB_MULTIHOP_BRIDGES.md)

## Call on musique/s1eval CH: SIGNAL

NEW@10 = 0.1731 over 2005 unlinked slots (rule: SIGNAL iff >= 0.1 with >= 200 slots).

## Part A: hub names as entity nodes, the question choosing

| dataset | carve | kind | unlinked | any key | only > DFMAX | HUB@5 | HUB@10 | HUB@20 | NEW@5 | NEW@10 | NEW@20 | NEW@10 found | C median | C p90 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| musique | s1sel | GP | 2663 | 0.5126 | 0.0 | 0.3875 | 0.4352 | 0.4615 | 0.1119 | 0.1179 | 0.1273 | 0.0398 | 5049 | 18739 |
| musique | s1sel | CH | 1447 | 0.718 | 0.0 | 0.548 | 0.6185 | 0.6558 | 0.1728 | 0.1811 | 0.2039 | 0.0581 | 4476 | 19012 |
| musique | s1eval | GP | 3957 | 0.5214 | 0.0 | 0.3571 | 0.4016 | 0.4377 | 0.0955 | 0.1145 | 0.1291 | 0.0364 | 3572 | 16308 |
| musique | s1eval | CH | 2005 | 0.7641 | 0.0 | 0.5117 | 0.5771 | 0.6359 | 0.1476 | 0.1731 | 0.201 | 0.0524 | 3774 | 16308 |
| 2wiki | s1sel | GP | 1687 | 0.2282 | 0.1761 | 0.0451 | 0.0462 | 0.0468 | 0.0047 | 0.0071 | 0.0053 | 0.0036 | 19858 | 100331 |
| 2wiki | s1eval | GP | 14265 | 0.2582 | 0.1933 | 0.0514 | 0.0528 | 0.0536 | 0.0107 | 0.0094 | 0.0068 | 0.0045 | 17793 | 104281 |
| hotpotqa | s1sel | GP | 297 | 0.3434 | 0.1852 | 0.1582 | 0.1582 | 0.1582 | 0.0135 | 0.0168 | 0.0067 | 0.0168 | 18394 | 83944 |
| hotpotqa | s1eval | GP | 1480 | 0.3169 | 0.1851 | 0.1318 | 0.1318 | 0.1318 | 0.0081 | 0.0041 | 0.0034 | 0.0041 | 18199 | 88451 |

## Part B: MuSiQue's unwritten bridges (unlinked chain slots)

### s1sel

| class | unlinked | share | all slots | HUB@10 from the earlier end | ans written in the earlier passage |
|---|---:|---:|---:|---:|---:|
| B1 | 862 | 0.5957 | 1180 | 0.5998 | 1.0 |
| B2 | 0 | None | 668 | None | None |
| B3u | 27 | 0.0187 | 27 | 0.4074 | 1.0 |
| B3w | 18 | 0.0124 | 18 | 0.1111 | 1.0 |
| B4 | 292 | 0.2018 | 370 | 0.1027 | 1.0 |
| B5 | 46 | 0.0318 | 53 | 0.0652 | 1.0 |
| B6 | 102 | 0.0705 | 108 | 0.451 | 1.0 |
| B7 | 100 | 0.0691 | 106 | 0.11 | 1.0 |

### s1eval

| class | unlinked | share | all slots | HUB@10 from the earlier end | ans written in the earlier passage |
|---|---:|---:|---:|---:|---:|
| B1 | 1255 | 0.6259 | 1795 | 0.5116 | 1.0 |
| B2 | 0 | None | 1178 | None | None |
| B3u | 2 | 0.001 | 2 | 0.0 | 1.0 |
| B3w | 2 | 0.001 | 8 | 0.0 | 1.0 |
| B4 | 432 | 0.2155 | 571 | 0.169 | 1.0 |
| B5 | 66 | 0.0329 | 120 | 0.1515 | 1.0 |
| B6 | 99 | 0.0494 | 145 | 0.2121 | 1.0 |
| B7 | 149 | 0.0743 | 168 | 0.1074 | 1.0 |

## Part C: how the chains and triplets are built (shares of slots each family links)

### musique/s1sel CH_by_hops

| group | n | W2_S | F0T | K | union |
|---|---:|---:|---:|---:|---:|
| 2 | 795 | 0.2138 | 0.1648 | 0.0302 | 0.3208 |
| 3 | 964 | 0.25 | 0.3071 | 0.1255 | 0.4616 |
| 4 | 771 | 0.249 | 0.3774 | 0.1102 | 0.4968 |

### musique/s1sel CH_by_relation

| group | n | W2_S | F0T | K | union |
|---|---:|---:|---:|---:|---:|
| (free text) | 1278 | 0.2433 | 0.2113 | 0.0775 | 0.3897 |
| located in the administrative territorial entity | 220 | 0.1636 | 0.4045 | 0.05 | 0.45 |
| place of birth | 190 | 0.2789 | 0.1474 | 0.0 | 0.3684 |
| country | 174 | 0.1149 | 0.0 | 0.0 | 0.1149 |
| performer | 142 | 0.4577 | 0.7746 | 0.3169 | 0.8239 |
| educated at | 67 | 0.0 | 0.4776 | 0.0 | 0.4776 |
| shares border with | 54 | 0.0556 | 0.7037 | 0.3333 | 0.7407 |
| author | 48 | 0.4375 | 1.0 | 0.7292 | 1.0 |
| place of death | 42 | 0.3571 | 0.1905 | 0.0 | 0.5 |
| country of citizenship | 30 | 0.1333 | 0.0 | 0.0 | 0.1333 |
| employer | 27 | 0.037 | 0.1111 | 0.0 | 0.1481 |
| contains administrative territorial entity | 18 | 0.0 | 0.8333 | 0.0 | 0.8333 |
| headquarters location | 17 | 0.2353 | 0.2941 | 0.0 | 0.2941 |
| director | 17 | 0.1765 | 0.8235 | 0.1765 | 0.8824 |
| producer | 14 | 0.4286 | 0.7143 | 0.1429 | 0.8571 |
| member of sports team | 12 | 0.5833 | 0.1667 | 0.0 | 0.6667 |
| country of origin | 11 | 0.0909 | 0.0 | 0.0 | 0.0909 |
| location | 11 | 0.0909 | 0.0 | 0.0 | 0.0909 |
| original language of film or tv show | 10 | 0.0 | 0.0 | 0.0 | 0.0 |
| located on terrain feature | 10 | 1.0 | 0.0 | 0.0 | 1.0 |
| military branch | 10 | 0.0 | 0.1 | 0.2 | 0.2 |
| named after | 10 | 0.0 | 0.0 | 0.0 | 0.0 |
| capital of | 9 | 0.6667 | 0.0 | 0.0 | 0.6667 |
| instance of | 8 | 0.0 | 0.0 | 0.0 | 0.0 |
| owned by | 7 | 0.0 | 0.2857 | 0.0 | 0.2857 |

### musique/s1eval CH_by_hops

| group | n | W2_S | F0T | K | union |
|---|---:|---:|---:|---:|---:|
| 2 | 1252 | 0.3498 | 0.6134 | 0.1701 | 0.7037 |
| 3 | 1520 | 0.1967 | 0.2086 | 0.075 | 0.3474 |
| 4 | 1215 | 0.284 | 0.2815 | 0.0881 | 0.4716 |

### musique/s1eval CH_by_relation

| group | n | W2_S | F0T | K | union |
|---|---:|---:|---:|---:|---:|
| (free text) | 2188 | 0.2838 | 0.2856 | 0.0864 | 0.4671 |
| located in the administrative territorial entity | 198 | 0.1313 | 0.5556 | 0.101 | 0.601 |
| performer | 192 | 0.4583 | 0.974 | 0.4375 | 0.974 |
| place of birth | 160 | 0.2313 | 0.2562 | 0.0063 | 0.3688 |
| country | 153 | 0.2026 | 0.1765 | 0.1634 | 0.2157 |
| headquarters location | 92 | 0.4457 | 0.3804 | 0.0326 | 0.7717 |
| place of death | 90 | 0.1778 | 0.1111 | 0.0222 | 0.2444 |
| author | 59 | 0.2881 | 0.8644 | 0.2034 | 0.8644 |
| operator | 58 | 0.0862 | 0.2069 | 0.0172 | 0.2069 |
| member of sports team | 47 | 0.0638 | 0.1277 | 0.0213 | 0.1277 |
| shares border with | 46 | 0.5435 | 0.913 | 0.0217 | 1.0 |
| creator | 42 | 0.3571 | 0.5238 | 0.2619 | 0.5238 |
| cast member | 36 | 0.1389 | 0.8889 | 0.1389 | 0.8889 |
| country of citizenship | 34 | 0.0294 | 0.0294 | 0.0 | 0.0294 |
| record label | 31 | 0.129 | 0.2581 | 0.129 | 0.2581 |
| owned by | 29 | 0.069 | 0.2759 | 0.1379 | 0.2759 |
| platform | 29 | 1.0 | 0.0 | 0.069 | 1.0 |
| manufacturer | 26 | 0.4231 | 0.1538 | 0.1154 | 0.5385 |
| winner | 25 | 0.08 | 0.08 | 0.08 | 0.12 |
| member of | 24 | 0.5417 | 0.2917 | 0.1667 | 0.5833 |
| composer | 24 | 0.0833 | 0.875 | 0.3333 | 0.875 |
| employer | 22 | 0.3636 | 0.5 | 0.0 | 0.5 |
| location of formation | 21 | 0.7143 | 0.1905 | 0.0 | 0.8095 |
| contains administrative territorial entity | 21 | 0.0 | 0.1905 | 0.0 | 0.1905 |
| spouse | 18 | 0.2222 | 0.7778 | 0.1667 | 0.7778 |

### 2wiki/s1sel GP_by_type

| group | n | W2_S | F0T | K | S | union |
|---|---:|---:|---:|---:|---:|---:|
| bridge_comparison | 1962 | 0.3043 | 0.1575 | 0.0423 | 0.3186 | 0.3318 |
| comparison | 362 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| compositional | 623 | 0.8844 | 0.4896 | 0.1445 | 0.9101 | 0.9807 |
| inference | 184 | 0.8967 | 0.7011 | 0.4457 | 0.8859 | 0.9891 |

### 2wiki/s1sel GP_by_first_relation

| group | n | W2_S | F0T | K | S | union |
|---|---:|---:|---:|---:|---:|---:|
| director | 2292 | 0.3892 | 0.2086 | 0.0454 | 0.4084 | 0.4263 |
| father | 186 | 0.8656 | 0.6882 | 0.4624 | 0.871 | 0.9839 |
| country | 116 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| spouse | 114 | 0.8509 | 0.5175 | 0.3421 | 0.886 | 0.9825 |
| country of citizenship | 68 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| composer | 64 | 0.9375 | 0.25 | 0.0312 | 0.8438 | 0.9531 |
| country of origin | 55 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| date of birth | 54 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| mother | 52 | 0.8846 | 0.75 | 0.3462 | 0.8846 | 0.9808 |
| performer | 49 | 0.9184 | 0.2653 | 0.0612 | 0.898 | 0.9796 |
| publication date | 36 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| date of death | 13 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| place of birth | 11 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| founded by | 8 | 0.875 | 0.75 | 0.125 | 0.875 | 0.875 |
| inception | 5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| sibling | 3 | 1.0 | 1.0 | 0.3333 | 1.0 | 1.0 |
| occupation | 3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| doctoral advisor | 1 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| child | 1 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 |

- 2wiki/s1sel bridge object: {"questions": 807, "written_in_subject": 692, "gold_titled_object": 655, "no_subject_passage": 111, "written_share": 0.9943, "titled_share": 0.9411}

### 2wiki/s1eval GP_by_type

| group | n | W2_S | F0T | K | S | union |
|---|---:|---:|---:|---:|---:|---:|
| bridge_comparison | 16506 | 0.3024 | 0.1501 | 0.0394 | 0.3163 | 0.3286 |
| comparison | 3040 | 0.0 | 0.0007 | 0.0003 | 0.0 | 0.001 |
| compositional | 5236 | 0.8585 | 0.6238 | 0.2701 | 0.8892 | 0.978 |
| inference | 1549 | 0.8121 | 0.7418 | 0.4777 | 0.876 | 0.98 |

### 2wiki/s1eval GP_by_first_relation

| group | n | W2_S | F0T | K | S | union |
|---|---:|---:|---:|---:|---:|---:|
| director | 18474 | 0.3654 | 0.2062 | 0.0479 | 0.3821 | 0.4001 |
| father | 2260 | 0.8239 | 0.7146 | 0.5097 | 0.8642 | 0.9765 |
| publication date | 1092 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| spouse | 867 | 0.8247 | 0.5709 | 0.3818 | 0.8708 | 0.9689 |
| date of birth | 840 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| mother | 696 | 0.8103 | 0.8218 | 0.3894 | 0.8764 | 0.9784 |
| performer | 426 | 0.8732 | 0.3122 | 0.115 | 0.9202 | 0.9859 |
| country of origin | 345 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| country | 316 | 0.0 | 0.0063 | 0.0032 | 0.0 | 0.0095 |
| composer | 262 | 0.9084 | 0.4427 | 0.0878 | 0.8969 | 0.9733 |
| country of citizenship | 203 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| date of death | 144 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| producer | 98 | 0.2755 | 0.1122 | 0.0102 | 0.2857 | 0.3163 |
| founded by | 71 | 0.8732 | 0.5352 | 0.338 | 0.831 | 0.9859 |
| inception | 64 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| child | 44 | 0.8636 | 0.6818 | 0.5682 | 0.8864 | 1.0 |
| publisher | 34 | 0.9118 | 0.3529 | 0.2647 | 0.8529 | 1.0 |
| sibling | 32 | 0.9375 | 0.75 | 0.5 | 0.9688 | 1.0 |
| creator | 18 | 0.9444 | 0.4444 | 0.2222 | 0.7778 | 1.0 |
| has part | 17 | 0.9412 | 0.8235 | 0.3529 | 0.7647 | 1.0 |
| editor | 12 | 0.9167 | 0.8333 | 0.3333 | 1.0 | 1.0 |
| presenter | 6 | 1.0 | 0.6667 | 0.5 | 0.6667 | 1.0 |
| occupation | 5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| manufacturer | 3 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| place of birth | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

- 2wiki/s1eval bridge object: {"questions": 6785, "written_in_subject": 5670, "gold_titled_object": 5318, "no_subject_passage": 1044, "written_share": 0.9876, "titled_share": 0.9263}

### hotpotqa/s1sel GP_by_type

| group | n | W2_S | F0T | K | S | union |
|---|---:|---:|---:|---:|---:|---:|
| bridge | 1205 | 0.7577 | 0.4763 | 0.2415 | 1.0 | 1.0 |
| comparison | 303 | 0.0 | 0.0033 | 0.0165 | 0.0 | 0.0198 |

### hotpotqa/s1eval GP_by_type

| group | n | W2_S | F0T | K | S | union |
|---|---:|---:|---:|---:|---:|---:|
| bridge | 5918 | 0.7254 | 0.4667 | 0.2506 | 1.0 | 1.0 |
| comparison | 1487 | 0.0007 | 0.002 | 0.002 | 0.0007 | 0.0047 |

