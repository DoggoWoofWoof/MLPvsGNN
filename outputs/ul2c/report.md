# UL2 section 12.6: carrying by budget

Verdict: **NONE_CARRIED** []. P* = 4997.1; bound 130/100 x |REC|; 2000 resamples, seed 0.

| form | carried | (a) | (b) | kept new pairs / new pairs, per dataset |
|---|---|---|---|---|
| H5 | False | True | False | musique 429,747/429,747; squad 60,681/60,681; hotpotqa 18,733,402/19,127,180; 2wiki 21,676,906/21,676,906; metaqa 82,016/82,016; webqsp 28,245,530/28,245,530 |
| HM | False | True | False | musique 265,292/265,292; squad 34,187/34,187; hotpotqa 10,458,416/10,458,416; 2wiki 11,998,902/11,998,902; metaqa 40,626/40,626; webqsp 9,325,326/9,325,326 |
| HK1 | False | True | False | musique 281,823/281,823; squad 38,713/38,713; hotpotqa 11,184,149/11,184,149; 2wiki 17,541,000/17,541,000; metaqa 35,663/35,663; webqsp 4,821,571/4,821,571 |
| NTc | False | True | False | musique 493,720/493,720; squad 72,853/72,853; hotpotqa 8,559,240/8,559,240; 2wiki 14,031,514/14,031,514; metaqa 83,377/83,377; webqsp 1,023,873/1,023,873 |
| NT1 | False | True | False | musique 417,521/417,521; squad 55,061/55,061; hotpotqa 15,483,311/15,483,311; 2wiki 24,449,948/24,449,948; metaqa 54,102/54,102; webqsp 4,781,062/4,781,062 |
| HK2 | False | True | False | musique 529,407/529,407; squad 72,729/72,729; hotpotqa 18,733,402/21,334,758; 2wiki 33,412,899/33,412,899; metaqa 67,184/67,184; webqsp 9,263,813/9,263,813 |
| H10 | False | True | False | musique 826,642/826,642; squad 116,928/116,928; hotpotqa 18,733,402/36,691,954; 2wiki 34,795,373/41,661,546; metaqa 107,641/162,921; webqsp 36,479,378/36,479,378 |

## Judged cells (s1eval), on what each form keeps

| form | cell | new | pairs added | price | gain (points) |
|---|---|---|---|---|---|
| H5 | musique/RK/GP | 93 | 429,747 | 4620.9 | +1.56 [+1.24, +1.89] ABOVE |
| H5 | musique/RK/CH | 86 | 429,747 | 4997.1 | +2.16 [+1.71, +2.62] ABOVE |
| H5 | hotpotqa/RK/GP | 110 | 18,733,402 | 170303.7 | +1.49 [+1.22, +1.77] ABOVE |
| H5 | hotpotqa/SRK/GP | 9 | 18,582,911 | 2064767.9 | +0.12 [+0.05, +0.20] ABOVE |
| H5 | 2wiki/RK/GP | 85 | 21,676,906 | 255022.4 | +0.32 [+0.26, +0.39] ABOVE |
| H5 | 2wiki/SRK/GP | 32 | 21,397,324 | 668666.4 | +0.12 [+0.08, +0.16] ABOVE |
| H5 | metaqa/SRK/GP | 931 | 82,016 | 88.1 | +2.49 [+2.22, +2.78] ABOVE |
| H5 | webqsp/SRK/GP | 187 | 28,245,180 | 151043.7 | +4.28 [+3.40, +5.23] ABOVE |
| HM | musique/RK/GP | 49 | 265,292 | 5414.1 | +0.82 [+0.60, +1.06] ABOVE |
| HM | musique/RK/CH | 48 | 265,292 | 5526.9 | +1.20 [+0.87, +1.55] ABOVE |
| HM | hotpotqa/RK/GP | 66 | 10,458,416 | 158460.8 | +0.89 [+0.69, +1.11] ABOVE |
| HM | hotpotqa/SRK/GP | 8 | 10,373,112 | 1296639.0 | +0.11 [+0.04, +0.19] ABOVE |
| HM | 2wiki/RK/GP | 57 | 11,998,902 | 210507.1 | +0.22 [+0.16, +0.27] ABOVE |
| HM | 2wiki/SRK/GP | 25 | 11,830,237 | 473209.5 | +0.09 [+0.06, +0.13] ABOVE |
| HM | metaqa/SRK/GP | 850 | 40,626 | 47.8 | +2.27 [+2.00, +2.56] ABOVE |
| HM | webqsp/SRK/GP | 124 | 9,325,116 | 75202.5 | +2.84 [+2.12, +3.73] ABOVE |
| HK1 | musique/RK/GP | 87 | 281,823 | 3239.3 | +1.46 [+1.15, +1.78] ABOVE |
| HK1 | musique/RK/CH | 77 | 281,823 | 3660.0 | +1.93 [+1.53, +2.35] ABOVE |
| HK1 | hotpotqa/RK/GP | 141 | 11,184,149 | 79320.2 | +1.90 [+1.59, +2.21] ABOVE |
| HK1 | hotpotqa/SRK/GP | 4 | 11,006,736 | 2751684.0 | +0.05 [+0.01, +0.11] ABOVE |
| HK1 | 2wiki/RK/GP | 107 | 17,541,000 | 163934.6 | +0.41 [+0.33, +0.48] ABOVE |
| HK1 | 2wiki/SRK/GP | 33 | 17,190,299 | 520918.2 | +0.13 [+0.08, +0.17] ABOVE |
| HK1 | metaqa/SRK/GP | 1549 | 35,660 | 23.0 | +4.14 [+3.69, +4.59] ABOVE |
| HK1 | webqsp/SRK/GP | 522 | 4,820,032 | 9233.8 | +11.95 [+10.69, +13.23] ABOVE |
| NTc | musique/RK/GP | 153 | 493,720 | 3226.9 | +2.57 [+2.17, +2.98] ABOVE |
| NTc | musique/RK/CH | 147 | 493,720 | 3358.6 | +3.69 [+3.13, +4.28] ABOVE |
| NTc | hotpotqa/RK/GP | 34 | 8,559,240 | 251742.4 | +0.46 [+0.31, +0.62] ABOVE |
| NTc | hotpotqa/SRK/GP | 2 | 8,497,699 | 4248849.5 | +0.03 [+0.00, +0.07] LEVEL |
| NTc | 2wiki/RK/GP | 43 | 14,031,514 | 326314.3 | +0.16 [+0.12, +0.21] ABOVE |
| NTc | 2wiki/SRK/GP | 24 | 13,919,481 | 579978.4 | +0.09 [+0.06, +0.13] ABOVE |
| NTc | metaqa/SRK/GP | 6181 | 83,376 | 13.5 | +16.51 [+15.46, +17.57] ABOVE |
| NTc | webqsp/SRK/GP | 151 | 1,023,435 | 6777.7 | +3.46 [+2.41, +4.63] ABOVE |
| NT1 | musique/RK/GP | 53 | 417,521 | 7877.8 | +0.89 [+0.66, +1.14] ABOVE |
| NT1 | musique/RK/CH | 53 | 417,521 | 7877.8 | +1.33 [+0.99, +1.71] ABOVE |
| NT1 | hotpotqa/RK/GP | 54 | 15,483,311 | 286728.0 | +0.73 [+0.55, +0.93] ABOVE |
| NT1 | hotpotqa/SRK/GP | 2 | 15,386,673 | 7693336.5 | +0.03 [+0.00, +0.07] LEVEL |
| NT1 | 2wiki/RK/GP | 58 | 24,449,948 | 421550.8 | +0.22 [+0.16, +0.28] ABOVE |
| NT1 | 2wiki/SRK/GP | 22 | 24,242,952 | 1101952.4 | +0.08 [+0.05, +0.12] ABOVE |
| NT1 | metaqa/SRK/GP | 1727 | 54,099 | 31.3 | +4.61 [+4.08, +5.18] ABOVE |
| NT1 | webqsp/SRK/GP | 178 | 4,780,207 | 26855.1 | +4.07 [+3.30, +4.89] ABOVE |
| HK2 | musique/RK/GP | 129 | 529,407 | 4103.9 | +2.16 [+1.79, +2.54] ABOVE |
| HK2 | musique/RK/CH | 116 | 529,407 | 4563.9 | +2.91 [+2.40, +3.43] ABOVE |
| HK2 | hotpotqa/RK/GP | 176 | 18,733,402 | 106439.8 | +2.38 [+2.04, +2.73] ABOVE |
| HK2 | hotpotqa/SRK/GP | 8 | 18,487,640 | 2310955.0 | +0.11 [+0.04, +0.19] ABOVE |
| HK2 | 2wiki/RK/GP | 177 | 33,412,899 | 188773.4 | +0.67 [+0.58, +0.77] ABOVE |
| HK2 | 2wiki/SRK/GP | 53 | 32,883,436 | 620442.2 | +0.20 [+0.15, +0.26] ABOVE |
| HK2 | metaqa/SRK/GP | 1903 | 67,179 | 35.3 | +5.08 [+4.55, +5.65] ABOVE |
| HK2 | webqsp/SRK/GP | 795 | 9,261,477 | 11649.7 | +18.20 [+16.49, +20.02] ABOVE |
| H10 | musique/RK/GP | 174 | 826,642 | 4750.8 | +2.92 [+2.50, +3.36] ABOVE |
| H10 | musique/RK/CH | 139 | 826,642 | 5947.1 | +3.49 [+2.93, +4.07] ABOVE |
| H10 | hotpotqa/RK/GP | 65 | 18,733,402 | 288206.2 | +0.88 [+0.68, +1.09] ABOVE |
| H10 | hotpotqa/SRK/GP | 6 | 18,597,783 | 3099630.5 | +0.08 [+0.03, +0.15] ABOVE |
| H10 | 2wiki/RK/GP | 102 | 34,795,373 | 341131.1 | +0.39 [+0.31, +0.46] ABOVE |
| H10 | 2wiki/SRK/GP | 38 | 34,421,354 | 905825.1 | +0.14 [+0.10, +0.19] ABOVE |
| H10 | metaqa/SRK/GP | 1225 | 107,641 | 87.9 | +3.27 [+2.91, +3.65] ABOVE |
| H10 | webqsp/SRK/GP | 300 | 36,478,893 | 121596.3 | +6.87 [+5.56, +8.30] ABOVE |

## Layout

- musique: union {'RK': 3388584}, ratio to REC {'RK': 1.0003}; 
- squad: union {'RK': 995869}, ratio to REC {'RK': 0.9982}; 
- hotpotqa: union {'RK': 62601137, 'SRK': 66832556}, ratio to REC {'RK': 1.0006, 'SRK': 1.0009}; 
- 2wiki: union {'RK': 115555310, 'SRK': 126541999}, ratio to REC {'RK': 0.9991, 'SRK': 0.9993}; 
- metaqa: union {'SRK': 349818}, ratio to REC {'SRK': 0.9941}; 
- webqsp: union {'SRK': 507129030}, ratio to REC {'SRK': 0.9995}; 
