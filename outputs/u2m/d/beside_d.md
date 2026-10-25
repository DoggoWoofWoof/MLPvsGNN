# U2m (d): the mass split by arrival type, beside

Declared in docs/U2M_POOLS_THREE_FAMILIES.md, section 13. It decides nothing: the parts are an input for U2n's models.

## metaqa (s1sel, 1497 questions, 6,563,380 rows; pick `own/kind_t/kt_cos:ada_f12_k0@0.85`, G = 0)

| walk | rows | restart | structural | ner | knn |
|---|---|---:|---:|---:|---:|
| A1st | gold | 0.1267 | 0.8733 | 0.0000 | 0.0000 |
| A1st | other | 0.2528 | 0.7472 | 0.0000 | 0.0000 |
| A1all | gold | 0.1356 | 0.6532 | 0.0186 | 0.1926 |
| A1all | other | 0.2294 | 0.4470 | 0.0333 | 0.2903 |
| Kst | gold | 0.0002 | 0.9998 | 0.0000 | 0.0000 |
| Kst | other | 0.0004 | 0.9996 | 0.0000 | 0.0000 |
| KTst | gold | 0.0001 | 0.9999 | 0.0000 | 0.0000 |
| KTst | other | 0.0004 | 0.9996 | 0.0000 | 0.0000 |
| KTall | gold | 0.0001 | 0.8739 | 0.0142 | 0.1118 |
| KTall | other | 0.0006 | 0.7435 | 0.0236 | 0.2324 |

Top (relation, direction) shares of the gold rows' structural mass (direction 1: the row is the stored source; 2: the destination):

- A1st: has_genre (2) 0.332; release_year (2) 0.145; starred_actors (1) 0.095; starred_actors (2) 0.075; has_tags (1) 0.063
- A1all: has_genre (2) 0.347; release_year (2) 0.152; starred_actors (1) 0.095; starred_actors (2) 0.066; written_by (1) 0.063
- Kst: starred_actors (1) 0.209; starred_actors (2) 0.125; written_by (1) 0.107; release_year (2) 0.098; has_genre (2) 0.088
- KTst: starred_actors (1) 0.213; starred_actors (2) 0.120; written_by (1) 0.107; release_year (2) 0.084; has_genre (2) 0.080
- KTall: starred_actors (1) 0.209; starred_actors (2) 0.115; written_by (1) 0.108; has_genre (2) 0.089; release_year (2) 0.084

## squad (s1sel, 1498 questions, 74,958 rows; pick `own/kind_t/kt_cos:ada_f12_k0@0.85`, G = 0)

| walk | rows | restart | structural | ner | knn |
|---|---|---:|---:|---:|---:|
| A1st | gold | 0.9210 | 0.0790 | 0.0000 | 0.0000 |
| A1st | other | 0.6666 | 0.3334 | 0.0000 | 0.0000 |
| A1all | gold | 0.8738 | 0.0256 | 0.0210 | 0.0796 |
| A1all | other | 0.6165 | 0.1302 | 0.0726 | 0.1807 |
| Kst | gold | - | - | - | - |
| Kst | other | - | - | - | - |
| KTst | gold | - | - | - | - |
| KTst | other | - | - | - | - |
| KTall | gold | - | - | - | - |
| KTall | other | - | - | - | - |

## musique (s1sel, 1534 questions, 8,341,012 rows; pick `own/kind_t/kt_cos:ada_f12_k0@0.85`, G = 0)

| walk | rows | restart | structural | ner | knn |
|---|---|---:|---:|---:|---:|
| A1st | gold | 0.8041 | 0.1959 | 0.0000 | 0.0000 |
| A1st | other | 0.3342 | 0.6658 | 0.0000 | 0.0000 |
| A1all | gold | 0.7163 | 0.0754 | 0.0995 | 0.1088 |
| A1all | other | 0.2977 | 0.2277 | 0.2225 | 0.2521 |
| Kst | gold | 0.5005 | 0.4995 | 0.0000 | 0.0000 |
| Kst | other | 0.0007 | 0.9993 | 0.0000 | 0.0000 |
| KTst | gold | 0.4550 | 0.5450 | 0.0000 | 0.0000 |
| KTst | other | 0.0006 | 0.9994 | 0.0000 | 0.0000 |
| KTall | gold | 0.4032 | 0.2708 | 0.1630 | 0.1631 |
| KTall | other | 0.0006 | 0.4030 | 0.2959 | 0.3006 |

## hotpotqa (s1sel, 1508 questions, 846,305 rows; pick `own/kind_t/kt_cos:ada_f12_k0@0.85`, G = 0)

| walk | rows | restart | structural | ner | knn |
|---|---|---:|---:|---:|---:|
| A1st | gold | 0.7826 | 0.2174 | 0.0000 | 0.0000 |
| A1st | other | 0.7817 | 0.2183 | 0.0000 | 0.0000 |
| A1all | gold | 0.7544 | 0.1287 | 0.0354 | 0.0816 |
| A1all | other | 0.6912 | 0.1143 | 0.0708 | 0.1237 |
| Kst | gold | 0.5645 | 0.4355 | 0.0000 | 0.0000 |
| Kst | other | 0.0062 | 0.9938 | 0.0000 | 0.0000 |
| KTst | gold | 0.5042 | 0.4958 | 0.0000 | 0.0000 |
| KTst | other | 0.0052 | 0.9948 | 0.0000 | 0.0000 |
| KTall | gold | 0.5090 | 0.2894 | 0.0795 | 0.1222 |
| KTall | other | 0.0052 | 0.4629 | 0.2160 | 0.3158 |

## 2wiki (s1sel, 1496 questions, 1,256,504 rows; pick `own/kind_t/kt_cos:ada_f12_k0@0.85`, G = 0)

| walk | rows | restart | structural | ner | knn |
|---|---|---:|---:|---:|---:|
| A1st | gold | 0.6610 | 0.3390 | 0.0000 | 0.0000 |
| A1st | other | 0.7840 | 0.2160 | 0.0000 | 0.0000 |
| A1all | gold | 0.6366 | 0.2311 | 0.0587 | 0.0735 |
| A1all | other | 0.7172 | 0.1058 | 0.0746 | 0.1024 |
| Kst | gold | 0.6015 | 0.3985 | 0.0000 | 0.0000 |
| Kst | other | 0.0009 | 0.9991 | 0.0000 | 0.0000 |
| KTst | gold | 0.5761 | 0.4239 | 0.0000 | 0.0000 |
| KTst | other | 0.0007 | 0.9993 | 0.0000 | 0.0000 |
| KTall | gold | 0.5703 | 0.2882 | 0.0734 | 0.0681 |
| KTall | other | 0.0008 | 0.4802 | 0.2661 | 0.2530 |

## webqsp (s1eval, 1503 questions, 7,146,196 rows; pick `own/kind_t/kt_cos:ada_f12_k0@0.85`, G = 0)

| walk | rows | restart | structural | ner | knn |
|---|---|---:|---:|---:|---:|
| A1st | gold | 0.1546 | 0.8454 | 0.0000 | 0.0000 |
| A1st | other | 0.3160 | 0.6840 | 0.0000 | 0.0000 |
| A1all | gold | 0.1637 | 0.6751 | 0.0578 | 0.1035 |
| A1all | other | 0.3108 | 0.4253 | 0.0750 | 0.1889 |
| Kst | gold | 0.0024 | 0.9976 | 0.0000 | 0.0000 |
| Kst | other | 0.0006 | 0.9994 | 0.0000 | 0.0000 |
| KTst | gold | 0.0022 | 0.9978 | 0.0000 | 0.0000 |
| KTst | other | 0.0006 | 0.9994 | 0.0000 | 0.0000 |
| KTall | gold | 0.0024 | 0.8561 | 0.0686 | 0.0729 |
| KTall | other | 0.0008 | 0.6510 | 0.0916 | 0.2566 |

Top (relation, direction) shares of the gold rows' structural mass (direction 1: the row is the stored source; 2: the destination):

- A1st: topic image (1) 0.067; topic article (1) 0.047; place lived location (2) 0.046; topic notable for (1) 0.039; film starring (1) 0.032
- A1all: topic image (1) 0.075; topic article (1) 0.053; place lived location (2) 0.051; topic notable for (1) 0.043; film starring (1) 0.033
- Kst: person profession (2) 0.063; location containedby (2) 0.053; location people born here (1) 0.025; performance film (2) 0.024; film starring (1) 0.023
- KTst: topic notable types (1) 0.044; ethnicity languages spoken (2) 0.039; statistical region population (1) 0.038; location containedby (2) 0.031; tourist attraction near travel destination (1) 0.026
- KTall: topic notable types (1) 0.045; ethnicity languages spoken (2) 0.039; location containedby (2) 0.037; topic image (1) 0.037; tourist attraction near travel destination (1) 0.028
