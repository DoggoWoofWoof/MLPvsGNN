# UL1 report (docs/UL1_UNIFIED_LINKS_ORACLE_CHECK.md): CARRIED F0T

Graph level, no model. A slot is linked when its pair is an edge of the union. Differences are paired over questions (ratio estimator), 95% interval of 2000 question resamples (seed 0). Domains (A1): RK = R + K, SRK = S + R + K, with R = W2_S + the form (ABL-U's W2_S on the KBs); pairs held per family under S > R > K. KB GP rows (questions with 2-10 golds) are beside.

## musique

nodes 117,534; family pairs: N_sp 800,278, K 265,366, W2_S 2,343,920, CN_all 931,069, F0T 973,013

| domain | none union | N_sp union | CN_all union | F0T union | F0T / N_sp | CN_all / N_sp |
|---|---:|---:|---:|---:|---:|---:|
| RK | 2,566,268 | 3,251,846 | 3,365,833 | 3,387,597 | 1.0417 | 1.0351 |

- RK, pairs held under S > R > K: none R 2,343,920 / K 222,348; N_sp R 3,067,861 / K 183,985; CN_all R 3,187,132 / K 178,701; F0T R 3,209,654 / K 177,943

| carve | domain | kind | slots | none share | N_sp share | CN_all share | F0T share | F0T - N_sp | CN_all - N_sp | N_sp - none | F0T - CN_all | price (F0T / CN_all / N_sp) |
|---|---|---|---:|---:|---:|---:|---:|:--|:--|:--|:--|:--|
| s1sel | RK | GP | 3,783 | 19.32% | 28.34% | 29.61% | 29.61% | +1.27 [+0.75, +1.83] ABOVE | +1.27 [+0.75, +1.83] ABOVE | +9.01 [+8.21, +9.88] ABOVE | +0.00 [+0.00, +0.00] LEVEL | 2111.4 / 2055.4 / 2010.5 |
| s1sel | RK | CH | 2,530 | 27.94% | 40.99% | 42.81% | 42.81% | +1.82 [+1.07, +2.63] ABOVE | +1.82 [+1.07, +2.63] ABOVE | +13.04 [+11.87, +14.30] ABOVE | +0.00 [+0.00, +0.00] LEVEL | 2184.4 / 2126.5 / 2077.5 |
| s1eval | RK | GP | 5,962 | 21.37% | 31.67% | 33.08% | 33.63% | +1.96 [+1.40, +2.51] ABOVE | +1.41 [+0.90, +1.93] ABOVE | +10.30 [+9.57, +11.08] ABOVE | +0.55 [+0.37, +0.75] ABOVE | 1123.6 / 1145.5 / 1116.6 |
| s1eval | RK | CH | 3,987 | 31.65% | 46.78% | 48.88% | 49.71% | +2.93 [+2.09, +3.72] ABOVE | +2.11 [+1.34, +2.86] ABOVE | +15.12 [+14.12, +16.20] ABOVE | +0.83 [+0.55, +1.13] ABOVE | 1140.7 / 1163.9 / 1136.9 |

- labels, s1sel RK GP, F0T's union (slots carrying / alone / per 10^6 pairs): R.name_comention 694/389/745.38; R.mention 626/347/267.07; K.knn 231/18/870.5; R.title_comention 37/0/882.13
- labels, s1sel RK GP, N_sp's union (slots carrying / alone / per 10^6 pairs): R.ner_spacy 657/341/820.96; R.mention 626/372/267.07; K.knn 231/26/870.5
- labels, s1sel RK CH, F0T's union (slots carrying / alone / per 10^6 pairs): R.name_comention 681/376/731.42; R.mention 603/324/257.26; K.knn 230/17/866.73; R.title_comention 37/0/882.13
- labels, s1sel RK CH, N_sp's union (slots carrying / alone / per 10^6 pairs): R.ner_spacy 646/330/807.22; R.mention 603/349/257.26; K.knn 230/25/866.73
- labels, s1eval RK GP, F0T's union (slots carrying / alone / per 10^6 pairs): R.name_comention 1326/698/1424.17; R.mention 1094/513/466.74; K.knn 434/31/1635.48; R.title_comention 112/33/2670.23
- labels, s1eval RK GP, N_sp's union (slots carrying / alone / per 10^6 pairs): R.ner_spacy 1200/614/1499.48; R.mention 1094/556/466.74; K.knn 434/56/1635.48
- labels, s1eval RK CH, F0T's union (slots carrying / alone / per 10^6 pairs): R.name_comention 1315/687/1412.36; R.mention 1082/501/461.62; K.knn 434/31/1635.48; R.title_comention 112/33/2670.23
- labels, s1eval RK CH, N_sp's union (slots carrying / alone / per 10^6 pairs): R.ner_spacy 1189/603/1485.73; R.mention 1082/544/461.62; K.knn 434/56/1635.48

MuSiQue s1sel, chain connected (every chain slot of the question linked, RK):

| form | 2 hops | 3 hops | 4 hops | all | minus N_sp |
|---|---:|---:|---:|---:|:--|
| none | 184/795 (23.14%) | 48/482 (9.96%) | 5/257 (1.95%) | 237/1534 (15.45%) |  |
| N_sp | 239/795 (30.06%) | 95/482 (19.71%) | 19/257 (7.39%) | 353/1534 (23.01%) |  |
| CN_all | 255/795 (32.08%) | 98/482 (20.33%) | 19/257 (7.39%) | 372/1534 (24.25%) | +1.24 [+0.59, +1.96] ABOVE |
| F0T | 255/795 (32.08%) | 98/482 (20.33%) | 19/257 (7.39%) | 372/1534 (24.25%) | +1.24 [+0.59, +1.96] ABOVE |

MuSiQue s1eval, chain connected (every chain slot of the question linked, RK):

| form | 2 hops | 3 hops | 4 hops | all | minus N_sp |
|---|---:|---:|---:|---:|:--|
| none | 528/1252 (42.17%) | 62/760 (8.16%) | 22/405 (5.43%) | 612/2417 (25.32%) |  |
| N_sp | 864/1252 (69.01%) | 101/760 (13.29%) | 35/405 (8.64%) | 1000/2417 (41.37%) |  |
| CN_all | 867/1252 (69.25%) | 107/760 (14.08%) | 33/405 (8.15%) | 1007/2417 (41.66%) | +0.29 [-0.46, +1.03] LEVEL |
| F0T | 881/1252 (70.37%) | 107/760 (14.08%) | 34/405 (8.4%) | 1022/2417 (42.28%) | +0.91 [+0.08, +1.70] ABOVE |

## squad

nodes 20,233; family pairs: N_sp 156,130, K 42,969, W2_S 878,200, CN_all 173,645, F0T 176,454

| domain | none union | N_sp union | CN_all union | F0T union | F0T / N_sp | CN_all / N_sp |
|---|---:|---:|---:|---:|---:|---:|
| RK | 888,311 | 986,232 | 997,604 | 997,652 | 1.0116 | 1.0115 |

- RK, pairs held under S > R > K: none R 878,200 / K 10,111; N_sp R 977,723 / K 8,509; CN_all R 989,323 / K 8,281; F0T R 989,371 / K 8,281

| carve | domain | kind | slots | none share | N_sp share | CN_all share | F0T share | F0T - N_sp | CN_all - N_sp | N_sp - none | F0T - CN_all | price (F0T / CN_all / N_sp) |
|---|---|---|---:|---:|---:|---:|---:|:--|:--|:--|:--|:--|
| s1sel | RK | GP | 0 | - | - | - | - | - | - | - | - | - |
| s1eval | RK | GP | 0 | - | - | - | - | - | - | - | - | - |


## hotpotqa

nodes 5,233,329; family pairs: N_sp 20,369,074, K 11,946,289, S 15,095,287, W2_S 32,159,229, CN_all 23,218,108, F0T 24,450,229

| domain | none union | N_sp union | CN_all union | F0T union | F0T / N_sp | CN_all / N_sp |
|---|---:|---:|---:|---:|---:|---:|
| RK | 42,847,105 | 59,814,434 | 61,793,991 | 62,565,030 | 1.0460 | 1.0331 |
| SRK | 47,346,662 | 64,080,356 | 66,022,204 | 66,773,876 | 1.0420 | 1.0303 |

- RK, pairs held under S > R > K: none R 32,159,229 / K 10,687,876; N_sp R 50,258,253 / K 9,556,181; CN_all R 52,432,584 / K 9,361,407; F0T R 53,243,304 / K 9,321,726
- SRK, pairs held under S > R > K: none S 15,095,287 / R 21,866,000 / K 10,385,375; N_sp S 15,095,287 / R 39,617,627 / K 9,367,442; CN_all S 15,095,287 / R 41,741,109 / K 9,185,808; F0T S 15,095,287 / R 42,527,421 / K 9,151,168

| carve | domain | kind | slots | none share | N_sp share | CN_all share | F0T share | F0T - N_sp | CN_all - N_sp | N_sp - none | F0T - CN_all | price (F0T / CN_all / N_sp) |
|---|---|---|---:|---:|---:|---:|---:|:--|:--|:--|:--|:--|
| s1sel | RK | GP | 1,508 | 64.92% | 67.18% | 68.17% | 68.63% | +1.46 [+0.66, +2.32] ABOVE | +0.99 [+0.27, +1.72] ABOVE | +2.25 [+1.53, +2.98] ABOVE | +0.46 [+0.13, +0.86] ABOVE | 352105.8 / 386671.1 / 499039.1 |
| s1sel | SRK | GP | 1,508 | 80.24% | 80.31% | 80.31% | 80.31% | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | +0.07 [+0.00, +0.20] LEVEL | +0.00 [+0.00, +0.00] LEVEL | 19427214.0 / 18675542.0 / 16733694.0 |
| s1eval | RK | GP | 7,405 | 62.78% | 65.13% | 65.46% | 65.90% | +0.77 [+0.46, +1.08] ABOVE | +0.32 [+0.05, +0.61] ABOVE | +2.35 [+2.01, +2.69] ABOVE | +0.45 [+0.30, +0.59] ABOVE | 85359.0 / 95691.3 / 97513.4 |
| s1eval | SRK | GP | 7,405 | 79.97% | 80.03% | 80.01% | 80.01% | -0.01 [-0.05, +0.03] LEVEL | -0.01 [-0.05, +0.03] LEVEL | +0.05 [+0.01, +0.11] ABOVE | +0.00 [+0.00, +0.00] LEVEL | 6475738.0 / 6225180.7 / 4183423.5 |

- labels, s1sel RK GP, F0T's union (slots carrying / alone / per 10^6 pairs): R.mention 913/369/28.39; R.name_comention 394/49/16.97; K.knn 296/47/24.78; R.title_comention 181/7/146.9
- labels, s1sel RK GP, N_sp's union (slots carrying / alone / per 10^6 pairs): R.mention 913/531/28.39; R.ner_spacy 312/34/15.32; K.knn 296/51/24.78
- labels, s1sel SRK GP, F0T's union (slots carrying / alone / per 10^6 pairs): S.hyperlink 1205/176/79.83; R.mention 913/0/28.39; R.name_comention 394/1/16.97; K.knn 296/5/24.78; R.title_comention 181/0/146.9
- labels, s1sel SRK GP, N_sp's union (slots carrying / alone / per 10^6 pairs): S.hyperlink 1205/198/79.83; R.mention 913/0/28.39; R.ner_spacy 312/1/15.32; K.knn 296/5/24.78
- labels, s1eval RK GP, F0T's union (slots carrying / alone / per 10^6 pairs): R.mention 4294/1705/133.52; R.name_comention 1940/198/83.56; K.knn 1486/197/124.39; R.title_comention 825/33/669.58
- labels, s1eval RK GP, N_sp's union (slots carrying / alone / per 10^6 pairs): R.mention 4294/2405/133.52; R.ner_spacy 1661/174/81.55; K.knn 1486/227/124.39
- labels, s1eval SRK GP, F0T's union (slots carrying / alone / per 10^6 pairs): S.hyperlink 5919/1045/392.11; R.mention 4294/0/133.52; R.name_comention 1940/3/83.56; K.knn 1486/3/124.39; R.title_comention 825/0/669.58
- labels, s1eval SRK GP, N_sp's union (slots carrying / alone / per 10^6 pairs): S.hyperlink 5919/1103/392.11; R.mention 4294/0/133.52; R.ner_spacy 1661/4/81.55; K.knn 1486/2/124.39

Beside: share of S's pairs held by the RK union: none 70.19%, N_sp 71.74%, CN_all 71.99%, F0T 72.12%

## 2wiki

nodes 5,989,847; family pairs: N_sp 34,067,241, K 13,643,063, S 28,208,857, W2_S 71,483,746, CN_all 38,242,751, F0T 39,863,302

| domain | none union | N_sp union | CN_all union | F0T union | F0T / N_sp | CN_all / N_sp |
|---|---:|---:|---:|---:|---:|---:|
| RK | 83,411,377 | 111,866,317 | 114,756,672 | 115,654,372 | 1.0339 | 1.0258 |
| SRK | 94,929,719 | 122,932,001 | 125,771,131 | 126,635,390 | 1.0301 | 1.0231 |

- RK, pairs held under S > R > K: none R 71,483,746 / K 11,927,631; N_sp R 101,245,772 / K 10,620,545; CN_all R 104,345,225 / K 10,411,447; F0T R 105,284,400 / K 10,369,972
- SRK, pairs held under S > R > K: none S 28,208,857 / R 55,330,052 / K 11,390,810; N_sp S 28,208,857 / R 84,459,003 / K 10,264,141; CN_all S 28,208,857 / R 87,493,236 / K 10,069,038; F0T S 28,208,857 / R 88,392,041 / K 10,034,492

| carve | domain | kind | slots | none share | N_sp share | CN_all share | F0T share | F0T - N_sp | CN_all - N_sp | N_sp - none | F0T - CN_all | price (F0T / CN_all / N_sp) |
|---|---|---|---:|---:|---:|---:|---:|:--|:--|:--|:--|:--|
| s1sel | RK | GP | 3,131 | 42.93% | 43.66% | 43.95% | 44.14% | +0.48 [+0.22, +0.77] ABOVE | +0.29 [+0.09, +0.53] ABOVE | +0.73 [+0.44, +1.05] ABOVE | +0.19 [+0.06, +0.36] ABOVE | 848499.9 / 979540.5 / 1237171.3 |
| s1sel | SRK | GP | 3,131 | 45.74% | 45.96% | 46.12% | 46.12% | +0.16 [+0.00, +0.35] LEVEL | +0.16 [+0.00, +0.35] LEVEL | +0.22 [+0.07, +0.40] ABOVE | +0.00 [+0.00, +0.00] LEVEL | 2642139.2 / 2570117.7 / 4000326.0 |
| s1eval | RK | GP | 26,331 | 42.06% | 43.37% | 43.57% | 43.80% | +0.43 [+0.33, +0.54] ABOVE | +0.20 [+0.11, +0.28] ABOVE | +1.31 [+1.17, +1.46] ABOVE | +0.24 [+0.18, +0.30] ABOVE | 70093.5 / 78757.0 / 82239.7 |
| s1eval | SRK | GP | 26,331 | 45.23% | 45.68% | 45.76% | 45.82% | +0.14 [+0.08, +0.21] ABOVE | +0.08 [+0.02, +0.14] ABOVE | +0.45 [+0.37, +0.54] ABOVE | +0.06 [+0.03, +0.10] ABOVE | 201946.9 / 220295.8 / 235313.3 |

- labels, s1sel RK GP, F0T's union (slots carrying / alone / per 10^6 pairs): R.mention 1313/593/18.37; R.name_comention 624/32/16.32; K.knn 255/11/18.69; R.title_comention 119/6/73.43
- labels, s1sel RK GP, N_sp's union (slots carrying / alone / per 10^6 pairs): R.mention 1313/706/18.37; R.ner_spacy 558/23/16.38; K.knn 255/15/18.69
- labels, s1sel SRK GP, F0T's union (slots carrying / alone / per 10^6 pairs): S.hyperlink 1355/62/48.03; R.mention 1313/32/18.37; R.name_comention 624/12/16.32; K.knn 255/5/18.69; R.title_comention 119/0/73.43
- labels, s1sel SRK GP, N_sp's union (slots carrying / alone / per 10^6 pairs): S.hyperlink 1355/72/48.03; R.mention 1313/40/18.37; R.ner_spacy 558/7/16.38; K.knn 255/8/18.69
- labels, s1eval RK GP, F0T's union (slots carrying / alone / per 10^6 pairs): R.mention 10744/4123/150.3; R.name_comention 5854/398/153.07; K.knn 2805/105/205.6; R.title_comention 1041/62/642.37
- labels, s1eval RK GP, N_sp's union (slots carrying / alone / per 10^6 pairs): R.mention 10744/5095/150.3; R.ner_spacy 5304/346/155.69; K.knn 2805/138/205.6
- labels, s1eval SRK GP, F0T's union (slots carrying / alone / per 10^6 pairs): S.hyperlink 11234/532/398.24; R.mention 10744/190/150.3; R.name_comention 5854/140/153.07; K.knn 2805/50/205.6; R.title_comention 1041/17/642.37
- labels, s1eval SRK GP, N_sp's union (slots carrying / alone / per 10^6 pairs): S.hyperlink 11234/608/398.24; R.mention 10744/277/150.3; R.ner_spacy 5304/119/155.69; K.knn 2805/64/205.6

Beside: share of S's pairs held by the RK union: none 59.17%, N_sp 60.77%, CN_all 60.95%, F0T 61.07%

## metaqa (KB, beside)

nodes 43,234; family pairs: N_sp 5,949, K 97,920, S 124,669, W2_S 245,144, CN_all 12,605, F0T 12,605

| domain | none union | N_sp union | CN_all union | F0T union | F0T / N_sp | CN_all / N_sp |
|---|---:|---:|---:|---:|---:|---:|
| SRK | 342,238 | 345,232 | 351,892 | 351,892 | 1.0193 | 1.0193 |

- SRK, pairs held under S > R > K: none S 124,669 / R 120,565 / K 97,004; N_sp S 124,669 / R 126,310 / K 94,253; CN_all S 124,669 / R 133,077 / K 94,146; F0T S 124,669 / R 133,077 / K 94,146

| carve | domain | kind | slots | none share | N_sp share | CN_all share | F0T share | F0T - N_sp | CN_all - N_sp | N_sp - none | F0T - CN_all | price (F0T / CN_all / N_sp) |
|---|---|---|---:|---:|---:|---:|---:|:--|:--|:--|:--|:--|
| s1sel | SRK | GP | 6,074 | 15.21% | 15.28% | 15.23% | 15.23% | -0.05 [-0.13, +0.00] LEVEL | -0.05 [-0.13, +0.00] LEVEL | +0.07 [+0.00, +0.17] LEVEL | +0.00 [+0.00, +0.00] LEVEL | 9654.0 / 9654.0 / 748.5 |
| s1eval | SRK | GP | 37,446 | 13.80% | 13.81% | 13.81% | 13.81% | +0.01 [+0.00, +0.02] LEVEL | +0.01 [+0.00, +0.02] LEVEL | +0.00 [+0.00, +0.01] LEVEL | +0.00 [+0.00, +0.00] LEVEL | 2413.5 / 2413.5 / 2994.0 |

- labels, s1sel SRK GP, F0T's union (slots carrying / alone / per 10^6 pairs): R.mention 718/677/2928.89; K.knn 212/195/2165.03; S.has_genre 33/0/2131.09; R.name_comention 12/1/952.0; S.release_year 2/0/119.65; R.title_comention 0/0/None
- labels, s1sel SRK GP, N_sp's union (slots carrying / alone / per 10^6 pairs): R.mention 718/677/2928.89; K.knn 212/198/2165.03; S.has_genre 33/0/2131.09; R.ner_spacy 13/4/2185.24; S.release_year 2/0/119.65
- labels, s1eval SRK GP, F0T's union (slots carrying / alone / per 10^6 pairs): R.mention 4280/4056/17459.13; K.knn 930/868/9497.55; S.has_genre 155/0/10009.69; R.name_comention 25/4/1983.34; S.release_year 24/0/1435.84; S.written_by 3/0/156.9; S.directed_by 1/0/62.67; S.in_language 1/0/291.89; R.title_comention 0/0/None
- labels, s1eval SRK GP, N_sp's union (slots carrying / alone / per 10^6 pairs): R.mention 4280/4055/17459.13; K.knn 930/882/9497.55; S.has_genre 155/0/10009.69; S.release_year 24/0/1435.84; R.ner_spacy 10/1/1680.95; S.written_by 3/0/156.9; S.directed_by 1/0/62.67; S.in_language 1/0/291.89

## webqsp (KB, beside)

nodes 2,592,894; family pairs: N_sp 2,994,802, K 6,470,520, S 5,882,501, W2_S 500,445,467, CN_all 995,434, F0T 997,211

| domain | none union | N_sp union | CN_all union | F0T union | F0T / N_sp | CN_all / N_sp |
|---|---:|---:|---:|---:|---:|---:|
| SRK | 506,654,997 | 508,556,193 | 507,397,362 | 507,398,750 | 0.9977 | 0.9977 |

- SRK, pairs held under S > R > K: none S 5,882,501 / R 494,579,148 / K 6,193,348; N_sp S 5,882,501 / R 496,802,391 / K 5,871,301; CN_all S 5,882,501 / R 495,463,573 / K 6,051,288; F0T S 5,882,501 / R 495,465,195 / K 6,051,054

| carve | domain | kind | slots | none share | N_sp share | CN_all share | F0T share | F0T - N_sp | CN_all - N_sp | N_sp - none | F0T - CN_all | price (F0T / CN_all / N_sp) |
|---|---|---|---:|---:|---:|---:|---:|:--|:--|:--|:--|:--|
| s1eval | SRK | GP | 4,369 | 20.55% | 20.55% | 20.78% | 20.78% | +0.23 [+0.02, +0.60] ABOVE | +0.23 [+0.02, +0.60] ABOVE | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | 74375.3 / 74236.5 / None |

- labels, s1eval SRK GP, F0T's union (slots carrying / alone / per 10^6 pairs): R.mention 733/436/1.46; K.knn 244/135/37.71; S.location containedby 98/0/642.11; R.name_comention 42/10/42.19; S.people profession specialization of 40/0/37593.98; S.profession specializations 34/0/17680.71; S.administrative area administrative children 23/0/4001.39; S.administrative area administrative parent 23/0/4234.17; S.administrative division country 23/0/6084.66; S.administrative division first level division of 20/0/19550.34; S.country administrative divisions 20/0/5367.69; S.country first level divisions 20/0/17699.12; and 102 more labels
- labels, s1eval SRK GP, N_sp's union (slots carrying / alone / per 10^6 pairs): R.mention 733/437/1.46; K.knn 244/146/37.71; S.location containedby 98/0/642.11; S.people profession specialization of 40/0/37593.98; S.profession specializations 34/0/17680.71; S.administrative area administrative children 23/0/4001.39; S.administrative area administrative parent 23/0/4234.17; S.administrative division country 23/0/6084.66; S.administrative division first level division of 20/0/19550.34; S.country administrative divisions 20/0/5367.69; S.country first level divisions 20/0/17699.12; S.film country 18/0/297.46; and 101 more labels

## The rule (section 6)

Bound: a form's union at most 1.3 times N_sp's. **CARRIED F0T.**

- F0T: holds everywhere
