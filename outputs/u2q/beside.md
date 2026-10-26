# U2q: the mass split by arrival type on the carried walk, beside

Declared in docs/U2Q_ARRIVAL_SPLIT_CARRIED.md. It decides nothing: the parts are an input for U2n's models. Each cell is the mean share of the type in a row's mass on the carried walk, with (d)'s filed share (U2e's walk at beta = 0) in brackets.

## metaqa (s1sel, 1497 questions, 6,541,747 rows; `own/kind_t/kt_cos:ada_f16_k300@0.85|G0|RWe@0.5|LW@0.5`, walk RWe@0.5)

| walk | rows | restart | structural | ner | knn |
|---|---|---:|---:|---:|---:|
| A1st | gold | 0.1135 (0.1267) | 0.8865 (0.8733) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| A1st | other | 0.2442 (0.2528) | 0.7558 (0.7472) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| A1all | gold | 0.1275 (0.1356) | 0.6677 (0.6532) | 0.0179 (0.0186) | 0.1870 (0.1926) |
| A1all | other | 0.2252 (0.2294) | 0.4462 (0.4470) | 0.0345 (0.0333) | 0.2942 (0.2903) |
| Kst | gold | 0.0002 (0.0002) | 0.9998 (0.9998) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| Kst | other | 0.0004 (0.0004) | 0.9996 (0.9996) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| KTst | gold | 0.0001 (0.0001) | 0.9999 (0.9999) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| KTst | other | 0.0004 (0.0004) | 0.9996 (0.9996) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| KTall | gold | 0.0001 (0.0001) | 0.8738 (0.8739) | 0.0140 (0.0142) | 0.1121 (0.1118) |
| KTall | other | 0.0006 (0.0006) | 0.7233 (0.7435) | 0.0267 (0.0236) | 0.2493 (0.2324) |

Top (relation, direction) shares of the gold rows' structural mass (direction 1: the row is the stored source; 2: the destination):

- A1st: has_genre (2) 0.283; release_year (2) 0.160; starred_actors (2) 0.093; starred_actors (1) 0.093; written_by (2) 0.065
- A1all: has_genre (2) 0.305; release_year (2) 0.171; starred_actors (1) 0.089; starred_actors (2) 0.081; written_by (2) 0.059
- Kst: starred_actors (1) 0.194; starred_actors (2) 0.145; release_year (2) 0.108; written_by (1) 0.097; written_by (2) 0.092
- KTst: starred_actors (1) 0.198; starred_actors (2) 0.136; directed_by (2) 0.099; release_year (2) 0.099; written_by (1) 0.098
- KTall: starred_actors (1) 0.191; starred_actors (2) 0.130; release_year (2) 0.100; written_by (1) 0.098; directed_by (2) 0.098

## squad (s1sel, 1498 questions, 371,553 rows; `own/kind_t/kt_cos:ada_f16_k300@0.85|G0|RWe@0.5|LW@0.5`, walk RWe@0.5 x LW@0.5)

| walk | rows | restart | structural | ner | knn |
|---|---|---:|---:|---:|---:|
| A1st | gold | 0.8873 (0.9210) | 0.1127 (0.0790) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| A1st | other | 0.6208 (0.6666) | 0.3792 (0.3334) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| A1all | gold | 0.8594 (0.8738) | 0.0377 (0.0256) | 0.0219 (0.0210) | 0.0809 (0.0796) |
| A1all | other | 0.5914 (0.6165) | 0.1619 (0.1302) | 0.0829 (0.0726) | 0.1638 (0.1807) |
| Kst | gold | - | - | - | - |
| Kst | other | - | - | - | - |
| KTst | gold | - | - | - | - |
| KTst | other | - | - | - | - |
| KTall | gold | - | - | - | - |
| KTall | other | - | - | - | - |

## musique (s1sel, 1534 questions, 8,471,028 rows; `own/kind_t/kt_cos:ada_f16_k300@0.85|G0|RWe@0.5|LW@0.5`, walk RWe@0.5 x LW@0.5)

| walk | rows | restart | structural | ner | knn |
|---|---|---:|---:|---:|---:|
| A1st | gold | 0.7794 (0.8041) | 0.2206 (0.1959) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| A1st | other | 0.3234 (0.3342) | 0.6766 (0.6658) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| A1all | gold | 0.7070 (0.7163) | 0.0883 (0.0754) | 0.0973 (0.0995) | 0.1074 (0.1088) |
| A1all | other | 0.2901 (0.2977) | 0.2339 (0.2277) | 0.2228 (0.2225) | 0.2532 (0.2521) |
| Kst | gold | 0.4901 (0.5005) | 0.5099 (0.4995) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| Kst | other | 0.0007 (0.0007) | 0.9993 (0.9993) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| KTst | gold | 0.4469 (0.4550) | 0.5531 (0.5450) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| KTst | other | 0.0007 (0.0006) | 0.9993 (0.9994) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| KTall | gold | 0.4009 (0.4032) | 0.2781 (0.2708) | 0.1584 (0.1630) | 0.1625 (0.1631) |
| KTall | other | 0.0006 (0.0006) | 0.3927 (0.4030) | 0.3010 (0.2959) | 0.3057 (0.3006) |

## hotpotqa (s1sel, 1508 questions, 1,077,147 rows; `own/kind_t/kt_cos:ada_f16_k300@0.85|G0|RWe@0.5|LW@0.5`, walk RWe@0.5 x LW@0.5)

| walk | rows | restart | structural | ner | knn |
|---|---|---:|---:|---:|---:|
| A1st | gold | 0.7322 (0.7826) | 0.2678 (0.2174) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| A1st | other | 0.7463 (0.7817) | 0.2537 (0.2183) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| A1all | gold | 0.7303 (0.7544) | 0.1588 (0.1287) | 0.0331 (0.0354) | 0.0778 (0.0816) |
| A1all | other | 0.6768 (0.6912) | 0.1217 (0.1143) | 0.0752 (0.0708) | 0.1262 (0.1237) |
| Kst | gold | 0.5492 (0.5645) | 0.4508 (0.4355) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| Kst | other | 0.0049 (0.0062) | 0.9951 (0.9938) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| KTst | gold | 0.4867 (0.5042) | 0.5133 (0.4958) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| KTst | other | 0.0045 (0.0052) | 0.9955 (0.9948) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| KTall | gold | 0.5032 (0.5090) | 0.2994 (0.2894) | 0.0771 (0.0795) | 0.1202 (0.1222) |
| KTall | other | 0.0042 (0.0052) | 0.4507 (0.4629) | 0.2261 (0.2160) | 0.3191 (0.3158) |

## 2wiki (s1sel, 1496 questions, 1,601,488 rows; `own/kind_t/kt_cos:ada_f16_k300@0.85|G0|RWe@0.5|LW@0.5`, walk RWe@0.5 x LW@0.5)

| walk | rows | restart | structural | ner | knn |
|---|---|---:|---:|---:|---:|
| A1st | gold | 0.6345 (0.6610) | 0.3655 (0.3390) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| A1st | other | 0.7513 (0.7840) | 0.2487 (0.2160) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| A1all | gold | 0.6225 (0.6366) | 0.2664 (0.2311) | 0.0450 (0.0587) | 0.0662 (0.0735) |
| A1all | other | 0.6996 (0.7172) | 0.1192 (0.1058) | 0.0762 (0.0746) | 0.1050 (0.1024) |
| Kst | gold | 0.5963 (0.6015) | 0.4037 (0.3985) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| Kst | other | 0.0007 (0.0009) | 0.9993 (0.9991) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| KTst | gold | 0.5579 (0.5761) | 0.4421 (0.4239) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| KTst | other | 0.0006 (0.0007) | 0.9994 (0.9993) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| KTall | gold | 0.5648 (0.5703) | 0.3118 (0.2882) | 0.0602 (0.0734) | 0.0632 (0.0681) |
| KTall | other | 0.0006 (0.0008) | 0.4759 (0.4802) | 0.2636 (0.2661) | 0.2598 (0.2530) |

## webqsp (s1eval, 1503 questions, 7,219,527 rows; `own/kind_t/kt_cos:ada_f16_k300@0.85|G0|RWe@0.5|LW@0.5`, walk RWe@0.5)

| walk | rows | restart | structural | ner | knn |
|---|---|---:|---:|---:|---:|
| A1st | gold | 0.1456 (0.1546) | 0.8544 (0.8454) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| A1st | other | 0.3107 (0.3160) | 0.6893 (0.6840) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| A1all | gold | 0.1573 (0.1637) | 0.6819 (0.6751) | 0.0600 (0.0578) | 0.1008 (0.1035) |
| A1all | other | 0.3088 (0.3108) | 0.4278 (0.4253) | 0.0748 (0.0750) | 0.1886 (0.1889) |
| Kst | gold | 0.0024 (0.0024) | 0.9976 (0.9976) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| Kst | other | 0.0006 (0.0006) | 0.9994 (0.9994) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| KTst | gold | 0.0022 (0.0022) | 0.9978 (0.9978) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| KTst | other | 0.0006 (0.0006) | 0.9994 (0.9994) | 0.0000 (0.0000) | 0.0000 (0.0000) |
| KTall | gold | 0.0023 (0.0024) | 0.8521 (0.8561) | 0.0713 (0.0686) | 0.0743 (0.0729) |
| KTall | other | 0.0008 (0.0008) | 0.6480 (0.6510) | 0.0916 (0.0916) | 0.2595 (0.2566) |

Top (relation, direction) shares of the gold rows' structural mass (direction 1: the row is the stored source; 2: the destination):

- A1st: topic image (1) 0.063; topic article (1) 0.045; place lived location (2) 0.039; topic notable for (1) 0.038; location containedby (2) 0.027
- A1all: topic image (1) 0.071; topic article (1) 0.050; place lived location (2) 0.043; topic notable for (1) 0.040; film starring (1) 0.026
- Kst: location containedby (2) 0.060; person profession (2) 0.051; ethnicity languages spoken (2) 0.028; topic notable types (1) 0.025; person place of birth (2) 0.021
- KTst: ethnicity languages spoken (2) 0.067; topic notable types (1) 0.044; location time zones (2) 0.034; location containedby (2) 0.030; statistical region population (1) 0.027
- KTall: ethnicity languages spoken (2) 0.074; topic notable types (1) 0.046; location time zones (2) 0.037; location containedby (2) 0.035; topic image (1) 0.030
