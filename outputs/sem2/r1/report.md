# SEM2 R1: link recovery under learned operators

Declared in docs/SEM2_EDGE_OPERATORS.md section 7. Written by zsem2.py report at 2026-11-24T18:16:00Z. MRR in points; intervals are paired 95% bootstrap intervals over the pooled test edges.

| dataset | arm | cosine MRR | rotation MRR | rotation - cosine | call | translation - cosine |
| --- | --- | --- | --- | --- | --- | --- |
| multihoprag | e3v | 17.36 | 24.12 | +6.76 [+5.16, +8.28] | LEARNS | +6.81 [+5.27, +8.32] ABOVE |
| multihoprag | frozen | 18.89 | 24.09 | +5.20 [+3.59, +6.81] | LEARNS | +3.94 [+2.24, +5.52] ABOVE |
| squad | e3v | 12.04 | 28.28 | +16.23 [+14.95, +17.54] | LEARNS | +13.17 [+12.03, +14.35] ABOVE |
| squad | frozen | 10.19 | 28.49 | +18.30 [+16.97, +19.68] | LEARNS | +14.46 [+13.27, +15.69] ABOVE |
| musique | e3v | 5.7 | 12.47 | +6.77 [+6.00, +7.60] | LEARNS | +3.99 [+3.33, +4.70] ABOVE |
| musique | frozen | 6.0 | 13.87 | +7.87 [+7.04, +8.74] | LEARNS | +5.03 [+4.34, +5.75] ABOVE |
| metaqa | e3v | 3.37 | 15.68 | +12.31 [+11.74, +12.92] | LEARNS | +13.54 [+12.90, +14.16] ABOVE |
| metaqa | frozen | 2.79 | 16.43 | +13.65 [+13.02, +14.28] | LEARNS | +15.80 [+15.13, +16.46] ABOVE |
| webqsp | e3v | 7.32 | 46.4 | +39.07 [+38.36, +39.75] | LEARNS | +39.88 [+39.14, +40.61] ABOVE |
| webqsp | frozen | 5.19 | 46.24 | +41.06 [+40.31, +41.75] | LEARNS | +42.31 [+41.56, +42.99] ABOVE |
| hotpotqa | e3v | 7.31 | 10.95 | +3.64 [+3.13, +4.17] | LEARNS | +3.07 [+2.56, +3.59] ABOVE |
| hotpotqa | frozen | 7.46 | 13.08 | +5.61 [+5.01, +6.25] | LEARNS | +4.77 [+4.18, +5.40] ABOVE |
| 2wiki | e3v | 6.74 | 9.36 | +2.62 [+2.11, +3.14] | LEARNS | +2.08 [+1.56, +2.58] ABOVE |
| 2wiki | frozen | 6.64 | 10.96 | +4.32 [+3.72, +4.87] | LEARNS | +3.35 [+2.80, +3.87] ABOVE |

**R1 per arm:** e3v HOLDS (7 of 7 LEARN); frozen HOLDS (7 of 7 LEARN)

**Expectations:** X1 held; X2 held; X3 held; X4 failed
