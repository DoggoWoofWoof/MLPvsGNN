# SEM2 R1h: hubs (report under amendment 8.9)

Declared in docs/SEM2_EDGE_OPERATORS.md section 8.9. Written by zsem2hr.py at 2026-11-24T10:39:26Z. MRR_T in points (each read group and target once); intervals are paired 95% bootstraps over those groups. Pooled MRR beside.

| dataset | arm | cosine | rotation | rotation+adj | HB | HB+adj | popularity | H1 (HB) | H2 (adj) | HB+adj vs R1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multihoprag | e3v | 16.74 | 23.43 | 16.15 | 23.19 | 16.96 | 2.13 | -0.24 [-1.05, +0.63] LEVEL | -7.28 [-9.03, -5.63] BELOW | -6.46 [-8.19, -4.76] BELOW |
| squad | e3v | 11.86 | 27.9 | 17.02 | 24.11 | 14.28 | 0.8 | -3.78 [-4.75, -2.84] BELOW | -10.88 [-12.05, -9.80] BELOW | -13.62 [-14.99, -12.29] BELOW |
| metaqa | e3v | 4.6 | 10.23 | 6.39 | 6.52 | 3.92 | 0.93 | -3.71 [-4.14, -3.30] BELOW | -3.84 [-4.24, -3.43] BELOW | -6.31 [-6.88, -5.77] BELOW |

| dataset | arm | pooled: rotation | rotation+adj | HB | HB+adj |
| --- | --- | --- | --- | --- | --- |
| multihoprag | e3v | 24.12 | 16.58 | 23.62 | 17.13 |
| squad | e3v | 28.28 | 17.32 | 24.29 | 14.68 |
| metaqa | e3v | 15.68 | 5.27 | 4.86 | 2.74 |

**Carry (8.9: ABOVE on MetaQA, BELOW on none; e3v, MRR_T):**
- HB: NOT_CARRIED (MetaQA BELOW, BELOW on ['squad', 'metaqa'], missing musique, webqsp, hotpotqa, 2wiki); 8.6's rule, superseded: INCOMPLETE (0 ABOVE)
- adjustment: NOT_CARRIED (MetaQA BELOW, BELOW on ['multihoprag', 'squad', 'metaqa'], missing musique, webqsp, hotpotqa, 2wiki); 8.6's rule, superseded: INCOMPLETE (0 ABOVE)
- HB + adjustment: NOT_CARRIED (MetaQA BELOW, BELOW on ['multihoprag', 'squad', 'metaqa'], missing musique, webqsp, hotpotqa, 2wiki); 8.6's rule, superseded: INCOMPLETE (0 ABOVE)

**R2 form:** none: the next hub step is declared from the per-label reads
