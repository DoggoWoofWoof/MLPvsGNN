# UL2 universal rules: the call (docs/UL2_LAYOUT_ON_E3V.md, section 12)

**BASE_OVER_BOUND** [] (P* = 4997.1 pairs per slot; bound 1.3 x |REC|; datasets musique, squad, hotpotqa, 2wiki, metaqa, webqsp; pending: multihop-rag (joins when MHR1 files its package)).

Beside: U-name (the KBs' F0T as filed) NONE_CARRIED []; MuSiQue's own chain as filed ['HK1', 'NTc'].

Graph level, no model. A slot is linked when its pair is in the union. Differences are paired over questions (ratio estimator), 95% interval of 2000 question resamples (seed 0), in points. s1eval decides.

Base over the bound: metaqa/SRK 2.599

| dataset | version | REC | B0 (U) | U / REC | U-name / REC | L2f / REC |
|---|---|---:|---:|---:|---:|---:|
| musique | RK | 3,387,597 | 3,388,584 | 1.0003 | 1.0003 | 1.1280 |
| squad | RK | 997,652 | 995,869 | 0.9982 | 0.9982 | 1.0600 |
| hotpotqa | RK | 62,565,030 | 62,601,137 | 1.0006 | 1.0006 | 1.3096 |
| hotpotqa | SRK | 66,773,876 | 66,832,556 | 1.0009 | 1.0009 | 1.2877 |
| 2wiki | RK | 115,654,372 | 115,555,310 | 0.9991 | 0.9991 | 1.1917 |
| 2wiki | SRK | 126,635,390 | 126,541,999 | 0.9993 | 0.9993 | 1.1729 |
| metaqa | SRK | 351,892 | 914,567 | 2.5990 | 0.9941 | 2.8459 |
| webqsp | SRK | 507,398,750 | 513,530,683 | 1.0121 | 0.9995 | 1.0712 |

## H5 (R.hub) over B0 + nothing: not carried (a True, b False, c False)

- price over P* in: hotpotqa/RK/GP, hotpotqa/SRK/GP, 2wiki/RK/GP, 2wiki/SRK/GP, webqsp/SRK/GP
- over the bound in: hotpotqa/RK, metaqa/SRK
- musique RK GP (5,962 slots): +93 slots, 429,747 pairs, price 4620.9, scale-free 8.133; +1.56 [+1.24, +1.89] ABOVE
- musique RK CH (3,987 slots): +86 slots, 429,747 pairs, price 4997.1, scale-free 5.881; +2.16 [+1.71, +2.62] ABOVE
- hotpotqa RK GP (7,405 slots): +114 slots, 19,127,180 pairs, price 167782.3, scale-free 19.858; +1.54 [+1.27, +1.84] ABOVE
- hotpotqa SRK GP (7,405 slots): +9 slots, 18,975,094 pairs, price 2108343.8, scale-free 233.808; +0.12 [+0.05, +0.20] ABOVE
- 2wiki RK GP (26,331 slots): +85 slots, 21,676,906 pairs, price 255022.4, scale-free 58.061; +0.32 [+0.26, +0.39] ABOVE
- 2wiki SRK GP (26,331 slots): +32 slots, 21,397,324 pairs, price 668666.4, scale-free 139.034; +0.12 [+0.08, +0.16] ABOVE
- metaqa SRK GP (37,446 slots): +931 slots, 86,189 pairs, price 92.6, scale-free 9.851; +2.49 [+2.22, +2.78] ABOVE
- webqsp SRK GP (4,369 slots): +198 slots, 28,421,963 pairs, price 143545.3, scale-free 1.236; +4.53 [+3.57, +5.68] ABOVE
- sizes: musique/RK +429,747 -> 1.1272; squad/RK +60,681 -> 1.059; hotpotqa/RK +19,127,180 -> 1.3063; hotpotqa/SRK +18,975,094 -> 1.285; 2wiki/RK +21,676,906 -> 1.1866; 2wiki/SRK +21,397,324 -> 1.1682; metaqa/SRK +86,189 -> 2.8439; webqsp/SRK +28,421,963 -> 1.0681

## HM (R.hub) over B0 + nothing: not carried (a True, b False, c False)

- price over P* in: musique/RK/GP, musique/RK/CH, hotpotqa/RK/GP, hotpotqa/SRK/GP, 2wiki/RK/GP, 2wiki/SRK/GP, webqsp/SRK/GP
- over the bound in: metaqa/SRK
- musique RK GP (5,962 slots): +49 slots, 265,292 pairs, price 5414.1, scale-free 9.529; +0.82 [+0.60, +1.06] ABOVE
- musique RK CH (3,987 slots): +48 slots, 265,292 pairs, price 5526.9, scale-free 6.505; +1.20 [+0.87, +1.55] ABOVE
- hotpotqa RK GP (7,405 slots): +66 slots, 10,458,416 pairs, price 158460.8, scale-free 18.755; +0.89 [+0.69, +1.11] ABOVE
- hotpotqa SRK GP (7,405 slots): +8 slots, 10,373,112 pairs, price 1296639.0, scale-free 143.793; +0.11 [+0.04, +0.19] ABOVE
- 2wiki RK GP (26,331 slots): +57 slots, 11,998,902 pairs, price 210507.1, scale-free 47.926; +0.22 [+0.16, +0.27] ABOVE
- 2wiki SRK GP (26,331 slots): +25 slots, 11,830,237 pairs, price 473209.5, scale-free 98.393; +0.09 [+0.06, +0.13] ABOVE
- metaqa SRK GP (37,446 slots): +850 slots, 44,272 pairs, price 52.1, scale-free 5.543; +2.27 [+2.00, +2.56] ABOVE
- webqsp SRK GP (4,369 slots): +136 slots, 9,472,422 pairs, price 69650.2, scale-free 0.6; +3.11 [+2.26, +4.30] ABOVE
- sizes: musique/RK +265,292 -> 1.0786; squad/RK +34,187 -> 1.0325; hotpotqa/RK +10,458,416 -> 1.1677; hotpotqa/SRK +10,373,112 -> 1.1562; 2wiki/RK +11,998,902 -> 1.1029; 2wiki/SRK +11,830,237 -> 1.0927; metaqa/SRK +44,272 -> 2.7248; webqsp/SRK +9,472,422 -> 1.0308

## HK1 (R.hub_key) over B0 + nothing: not carried (a True, b False, c False)

- price over P* in: hotpotqa/RK/GP, hotpotqa/SRK/GP, 2wiki/RK/GP, 2wiki/SRK/GP, webqsp/SRK/GP
- over the bound in: metaqa/SRK
- musique RK GP (5,962 slots): +87 slots, 281,823 pairs, price 3239.3, scale-free 5.701; +1.46 [+1.15, +1.78] ABOVE
- musique RK CH (3,987 slots): +77 slots, 281,823 pairs, price 3660.0, scale-free 4.308; +1.93 [+1.53, +2.35] ABOVE
- hotpotqa RK GP (7,405 slots): +141 slots, 11,184,149 pairs, price 79320.2, scale-free 9.388; +1.90 [+1.59, +2.21] ABOVE
- hotpotqa SRK GP (7,405 slots): +4 slots, 11,006,736 pairs, price 2751684.0, scale-free 305.153; +0.05 [+0.01, +0.11] ABOVE
- 2wiki RK GP (26,331 slots): +107 slots, 17,541,000 pairs, price 163934.6, scale-free 37.323; +0.41 [+0.33, +0.48] ABOVE
- 2wiki SRK GP (26,331 slots): +33 slots, 17,190,299 pairs, price 520918.2, scale-free 108.313; +0.13 [+0.08, +0.17] ABOVE
- metaqa SRK GP (37,446 slots): +1549 slots, 37,509 pairs, price 24.2, scale-free 2.577; +4.14 [+3.69, +4.59] ABOVE
- webqsp SRK GP (4,369 slots): +531 slots, 4,909,315 pairs, price 9245.4, scale-free 0.08; +12.15 [+10.90, +13.43] ABOVE
- sizes: musique/RK +281,823 -> 1.0835; squad/RK +38,713 -> 1.037; hotpotqa/RK +11,184,149 -> 1.1793; hotpotqa/SRK +11,006,736 -> 1.1657; 2wiki/RK +17,541,000 -> 1.1508; 2wiki/SRK +17,190,299 -> 1.135; metaqa/SRK +37,509 -> 2.7056; webqsp/SRK +4,909,315 -> 1.0218

## NTc (R.written) over B0 + nothing: not carried (a True, b False, c False)

- price over P* in: hotpotqa/RK/GP, hotpotqa/SRK/GP, 2wiki/RK/GP, 2wiki/SRK/GP, webqsp/SRK/GP
- over the bound in: metaqa/SRK
- musique RK GP (5,962 slots): +153 slots, 493,720 pairs, price 3226.9, scale-free 5.679; +2.57 [+2.17, +2.98] ABOVE
- musique RK CH (3,987 slots): +147 slots, 493,720 pairs, price 3358.6, scale-free 3.953; +3.69 [+3.13, +4.28] ABOVE
- hotpotqa RK GP (7,405 slots): +34 slots, 8,559,240 pairs, price 251742.4, scale-free 29.795; +0.46 [+0.31, +0.62] ABOVE
- hotpotqa SRK GP (7,405 slots): +2 slots, 8,497,699 pairs, price 4248849.5, scale-free 471.183; +0.03 [+0.00, +0.07] LEVEL
- 2wiki RK GP (26,331 slots): +43 slots, 14,031,514 pairs, price 326314.3, scale-free 74.292; +0.16 [+0.12, +0.21] ABOVE
- 2wiki SRK GP (26,331 slots): +24 slots, 13,919,481 pairs, price 579978.4, scale-free 120.594; +0.09 [+0.06, +0.13] ABOVE
- metaqa SRK GP (37,446 slots): +688 slots, 74,948 pairs, price 108.9, scale-free 11.592; +1.84 [+1.61, +2.09] ABOVE
- webqsp SRK GP (4,369 slots): +9 slots, 906,241 pairs, price 100693.4, scale-free 0.867; +0.21 [+0.07, +0.36] ABOVE
- sizes: musique/RK +493,720 -> 1.146; squad/RK +72,853 -> 1.0712; hotpotqa/RK +8,559,240 -> 1.1374; hotpotqa/SRK +8,497,699 -> 1.1281; 2wiki/RK +14,031,514 -> 1.1205; 2wiki/SRK +13,919,481 -> 1.1092; metaqa/SRK +74,948 -> 2.812; webqsp/SRK +906,241 -> 1.0139

## NT1 (R.written) over B0 + nothing: not carried (a True, b False, c False)

- price over P* in: musique/RK/GP, musique/RK/CH, hotpotqa/RK/GP, hotpotqa/SRK/GP, 2wiki/RK/GP, 2wiki/SRK/GP, webqsp/SRK/GP
- over the bound in: metaqa/SRK
- musique RK GP (5,962 slots): +53 slots, 417,521 pairs, price 7877.8, scale-free 13.864; +0.89 [+0.66, +1.14] ABOVE
- musique RK CH (3,987 slots): +53 slots, 417,521 pairs, price 7877.8, scale-free 9.272; +1.33 [+0.99, +1.71] ABOVE
- hotpotqa RK GP (7,405 slots): +54 slots, 15,483,311 pairs, price 286728.0, scale-free 33.936; +0.73 [+0.55, +0.93] ABOVE
- hotpotqa SRK GP (7,405 slots): +2 slots, 15,386,673 pairs, price 7693336.5, scale-free 853.165; +0.03 [+0.00, +0.07] LEVEL
- 2wiki RK GP (26,331 slots): +58 slots, 24,449,948 pairs, price 421550.8, scale-free 95.974; +0.22 [+0.16, +0.28] ABOVE
- 2wiki SRK GP (26,331 slots): +22 slots, 24,242,952 pairs, price 1101952.4, scale-free 229.126; +0.08 [+0.05, +0.12] ABOVE
- metaqa SRK GP (37,446 slots): +1727 slots, 54,131 pairs, price 31.3, scale-free 3.335; +4.61 [+4.08, +5.18] ABOVE
- webqsp SRK GP (4,369 slots): +178 slots, 4,791,660 pairs, price 26919.4, scale-free 0.232; +4.07 [+3.30, +4.89] ABOVE
- sizes: musique/RK +417,521 -> 1.1235; squad/RK +55,061 -> 1.0534; hotpotqa/RK +15,483,311 -> 1.2481; hotpotqa/SRK +15,386,673 -> 1.2313; 2wiki/RK +24,449,948 -> 1.2105; 2wiki/SRK +24,242,952 -> 1.1907; metaqa/SRK +54,131 -> 2.7528; webqsp/SRK +4,791,660 -> 1.0215

## HK2 (R.hub_key) over B0 + nothing: not carried (a True, b False, c False)

- price over P* in: hotpotqa/RK/GP, hotpotqa/SRK/GP, 2wiki/RK/GP, 2wiki/SRK/GP, webqsp/SRK/GP
- over the bound in: hotpotqa/RK, hotpotqa/SRK, metaqa/SRK
- musique RK GP (5,962 slots): +129 slots, 529,407 pairs, price 4103.9, scale-free 7.223; +2.16 [+1.79, +2.54] ABOVE
- musique RK CH (3,987 slots): +116 slots, 529,407 pairs, price 4563.9, scale-free 5.371; +2.91 [+2.40, +3.43] ABOVE
- hotpotqa RK GP (7,405 slots): +212 slots, 21,334,758 pairs, price 100635.7, scale-free 11.911; +2.86 [+2.48, +3.24] ABOVE
- hotpotqa SRK GP (7,405 slots): +9 slots, 21,065,275 pairs, price 2340586.1, scale-free 259.563; +0.12 [+0.05, +0.20] ABOVE
- 2wiki RK GP (26,331 slots): +177 slots, 33,412,899 pairs, price 188773.4, scale-free 42.978; +0.67 [+0.58, +0.77] ABOVE
- 2wiki SRK GP (26,331 slots): +53 slots, 32,883,436 pairs, price 620442.2, scale-free 129.007; +0.20 [+0.15, +0.26] ABOVE
- metaqa SRK GP (37,446 slots): +1903 slots, 70,221 pairs, price 36.9, scale-free 3.927; +5.08 [+4.55, +5.65] ABOVE
- webqsp SRK GP (4,369 slots): +808 slots, 9,409,884 pairs, price 11645.9, scale-free 0.1; +18.49 [+16.70, +20.32] ABOVE
- sizes: musique/RK +529,407 -> 1.1566; squad/RK +72,729 -> 1.0711; hotpotqa/RK +21,334,758 -> 1.3416; hotpotqa/SRK +21,065,275 -> 1.3164; 2wiki/RK +33,412,899 -> 1.288; 2wiki/SRK +32,883,436 -> 1.2589; metaqa/SRK +70,221 -> 2.7986; webqsp/SRK +9,409,884 -> 1.0306

## H10 (R.hub) over B0 + nothing: not carried (a True, b False, c False)

- price over P* in: musique/RK/CH, hotpotqa/RK/GP, hotpotqa/SRK/GP, 2wiki/RK/GP, 2wiki/SRK/GP, webqsp/SRK/GP
- over the bound in: hotpotqa/RK, hotpotqa/SRK, 2wiki/RK, 2wiki/SRK, metaqa/SRK
- musique RK GP (5,962 slots): +174 slots, 826,642 pairs, price 4750.8, scale-free 8.361; +2.92 [+2.50, +3.36] ABOVE
- musique RK CH (3,987 slots): +139 slots, 826,642 pairs, price 5947.1, scale-free 6.999; +3.49 [+2.93, +4.07] ABOVE
- hotpotqa RK GP (7,405 slots): +167 slots, 36,691,954 pairs, price 219712.3, scale-free 26.004; +2.26 [+1.93, +2.59] ABOVE
- hotpotqa SRK GP (7,405 slots): +12 slots, 36,465,908 pairs, price 3038825.7, scale-free 336.996; +0.16 [+0.08, +0.26] ABOVE
- 2wiki RK GP (26,331 slots): +150 slots, 41,661,546 pairs, price 277743.6, scale-free 63.234; +0.57 [+0.48, +0.66] ABOVE
- 2wiki SRK GP (26,331 slots): +44 slots, 41,243,947 pairs, price 937362.4, scale-free 194.904; +0.17 [+0.12, +0.22] ABOVE
- metaqa SRK GP (37,446 slots): +1476 slots, 168,883 pairs, price 114.4, scale-free 12.176; +3.94 [+3.53, +4.38] ABOVE
- webqsp SRK GP (4,369 slots): +314 slots, 36,747,900 pairs, price 117031.5, scale-free 1.008; +7.19 [+5.77, +8.80] ABOVE
- sizes: musique/RK +826,642 -> 1.2443; squad/RK +116,928 -> 1.1154; hotpotqa/RK +36,691,954 -> 1.587; hotpotqa/SRK +36,465,908 -> 1.547; 2wiki/RK +41,661,546 -> 1.3594; 2wiki/SRK +41,243,947 -> 1.325; metaqa/SRK +168,883 -> 3.0789; webqsp/SRK +36,747,900 -> 1.0845

