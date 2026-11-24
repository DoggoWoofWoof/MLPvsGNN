# SEM2 R1h part A: R1's ranks grouped by target

Declared in docs/SEM2_EDGE_OPERATORS.md section 8. Written by zsem2h.py hubread at 2026-11-24T09:28:46Z. MRR in points. MRR_T: each (read group, target) counts once. H0: rotation - cosine on MRR_T, paired bootstrap over groups.

| dataset arm | cosine MRR / MRR_T | rotation MRR / MRR_T | pooled gain (R1's call) | MRR_T gain (H0) |
| --- | --- | --- | --- | --- |
| multihoprag e3v | 17.36 / 16.74 | 24.12 / 23.43 | +6.76 [+5.16, +8.28] LEARNS | +6.69 [+4.81, +8.56] LEARNS |
| multihoprag frozen | 18.89 / 18.38 | 24.09 / 23.55 | +5.20 [+3.59, +6.81] LEARNS | +5.17 [+3.20, +7.05] LEARNS |
| squad e3v | 12.04 / 11.86 | 28.28 / 27.9 | +16.23 [+14.95, +17.54] LEARNS | +16.04 [+14.64, +17.41] LEARNS |
| squad frozen | 10.19 / 9.92 | 28.49 / 28.54 | +18.30 [+16.97, +19.68] LEARNS | +18.61 [+17.22, +20.00] LEARNS |
| musique e3v | 5.7 / 5.81 | 12.47 / 12.1 | +6.77 [+6.00, +7.60] LEARNS | +6.29 [+5.50, +7.14] LEARNS |
| musique frozen | 6.0 / 6.14 | 13.87 / 13.52 | +7.87 [+7.04, +8.74] LEARNS | +7.37 [+6.52, +8.23] LEARNS |
| metaqa e3v | 3.37 / 4.6 | 15.68 / 10.23 | +12.31 [+11.74, +12.92] LEARNS | +5.63 [+5.11, +6.14] LEARNS |
| metaqa frozen | 2.79 / 3.88 | 16.43 / 10.89 | +13.65 [+13.02, +14.28] LEARNS | +7.01 [+6.48, +7.56] LEARNS |
| webqsp e3v | 7.32 / 13.87 | 46.4 / 49.3 | +39.07 [+38.36, +39.75] LEARNS | +35.43 [+34.42, +36.38] LEARNS |
| webqsp frozen | 5.19 / 9.87 | 46.24 / 49.89 | +41.06 [+40.31, +41.75] LEARNS | +40.02 [+38.98, +41.05] LEARNS |
| hotpotqa e3v | 7.31 / 7.69 | 10.95 / 11.3 | +3.64 [+3.13, +4.17] LEARNS | +3.61 [+3.06, +4.14] LEARNS |
