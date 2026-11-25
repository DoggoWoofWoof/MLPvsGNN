# SEM2 R2 report

Declared in docs/SEM2_EDGE_OPERATORS.md section 9. e3v verdict: **PARTIAL**; ROOM (C >= 1.0 pt): none.

| arm/dataset | T1 trav - base | T2 trav - trav0 | C oracle - trav | reach | lambda trav/trav0 | ADDS |
| --- | --- | --- | --- | --- | --- | --- |
| e3v/multihoprag | +1.84 [+1.01, +2.65] ABOVE | +0.18 [+0.00, +0.38] LEVEL | +0.00 [+0.00, +0.00] LEVEL | 0.997 | 0.0/0.0 | False |
| e3v/squad | +0.28 [+0.04, +0.51] ABOVE | +0.11 [+0.03, +0.19] ABOVE | -0.01 [-0.03, +0.02] LEVEL | 0.813 | 2.0/1.0 | True |
| e3v/musique | +1.22 [+0.76, +1.67] ABOVE | -0.22 [-0.37, -0.09] BELOW | +0.03 [-0.06, +0.12] LEVEL | 0.891 | 0.0/0.0 | False |
| e3v/metaqa | +0.11 [+0.04, +0.19] ABOVE | +0.06 [+0.02, +0.11] ABOVE | +0.02 [+0.00, +0.05] LEVEL | 0.548 | 0.0/0.0 | True |
| e3v/webqsp | +0.38 [+0.09, +0.72] ABOVE | +0.00 [+0.00, +0.01] LEVEL | -0.02 [-0.07, +0.00] LEVEL | 0.925 | 0.0/0.0 | False |
| e3v/hotpotqa | +0.97 [+0.71, +1.22] ABOVE | +0.08 [-0.02, +0.18] LEVEL | -0.02 [-0.08, +0.04] LEVEL | 0.936 | 0.25/0.0 | False |
| e3v/2wiki | +0.65 [+0.51, +0.80] ABOVE | +0.03 [-0.02, +0.08] LEVEL | +0.04 [-0.00, +0.08] LEVEL | 0.866 | 0.25/0.25 | False |
| frozen/multihoprag | +2.04 [+1.21, +2.87] ABOVE | +0.21 [-0.05, +0.50] LEVEL | +0.00 [+0.00, +0.00] LEVEL | 0.998 | 2.0/1.0 | False |
| frozen/squad | +0.45 [+0.21, +0.67] ABOVE | +0.08 [-0.02, +0.18] LEVEL | +0.01 [-0.03, +0.04] LEVEL | 0.824 | 2.0/0.0 | False |
| frozen/musique | +1.22 [+0.77, +1.67] ABOVE | +0.13 [-0.08, +0.34] LEVEL | +0.05 [+0.01, +0.11] ABOVE | 0.894 | 2.0/0.25 | False |
| frozen/metaqa | +0.22 [+0.15, +0.30] ABOVE | +0.01 [-0.04, +0.07] LEVEL | +0.00 [+0.00, +0.00] LEVEL | 0.545 | 2.0/0.25 | False |

Expectations: X12 T1 ABOVE on MuSiQue: True; X13 T1 not ABOVE on SQuAD: False; X14 ROOM on MetaQA: False
