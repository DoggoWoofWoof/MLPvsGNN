# SEM2 R1c: the calibrated adjustment

Declared in docs/SEM2_EDGE_OPERATORS.md section 8.10. Written by zsem2hc.py report at 2026-11-24T11:54:31Z. MRR_T in points (each read group and target once); paired 95% bootstraps over those groups. Betas chosen per read group on R1's dev sample.

| dataset | arm | cosine | cosine cal | rotation | rotation cal | H3 (MRR_T) | H3 on pooled | rotation betas > 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multihoprag | e3v | 16.74 | 16.74 | 23.43 | 23.43 | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | none |
| multihoprag | frozen | 18.38 | 18.38 | 23.55 | 23.55 | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | none |
| squad | e3v | 11.86 | 11.86 | 27.9 | 27.9 | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | none |
| squad | frozen | 9.92 | 9.92 | 28.54 | 28.54 | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | none |
| musique | e3v | 5.81 | 5.81 | 12.1 | 12.1 | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | none |
| musique | frozen | 6.14 | 6.14 | 13.52 | 13.52 | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | none |
| metaqa | e3v | 4.6 | 4.6 | 10.23 | 10.21 | -0.02 [-0.03, -0.00] BELOW | -0.02 [-0.03, -0.00] BELOW | {'S.written_by': 0.25} |
| metaqa | frozen | 3.88 | 3.88 | 10.89 | 10.89 | +0.00 [+0.00, +0.00] LEVEL | +0.00 [+0.00, +0.00] LEVEL | none |

**Carry (8.10, as 8.9: ABOVE on MetaQA, BELOW on none; e3v, MRR_T):** NOT_CARRIED (MetaQA BELOW, BELOW on ['metaqa'], missing webqsp, hotpotqa, 2wiki)

**Expectations:** X7 True; X8 False
