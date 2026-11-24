# SEM2 R1k: the hub ceiling

Declared in docs/SEM2_EDGE_OPERATORS.md section 8.11. Written by zsem2hk.py report at 2026-11-24T15:40:29Z. The ceiling ranks each test edge among the nodes no more popular than its true target (deg_L), the best any non-decreasing degree adjustment can do. MRR_T in points; paired 95% bootstraps over (read group, target).

| dataset | arm | rotation | ceiling | C | verdict | largest per-group C | crowded % | cosine C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multihoprag | e3v | 23.43 | 28.35 | +4.93 [+4.06, +5.81] | ROOM | R.name_comention +4.93 | 74.5 | +4.55 [+3.57, +5.48] |
| squad | e3v | 27.9 | 32.76 | +4.86 [+4.33, +5.43] | ROOM | R.title_comention +10.16 | 62.48 | +3.66 [+3.20, +4.16] |
| musique | e3v | 12.1 | 14.27 | +2.16 [+1.85, +2.50] | ROOM | R.mention +2.43 | 77.5 | +1.22 [+0.99, +1.47] |
| metaqa | e3v | 10.23 | 11.13 | +0.90 [+0.78, +1.02] | NO_ROOM | S.has_genre +11.72 | 79.06 | +0.12 [+0.08, +0.16] |
| webqsp | e3v | 49.3 | 49.97 | +0.68 [+0.57, +0.78] | NO_ROOM | S.person nationality +8.26 | 37.45 | +0.16 [+0.11, +0.21] |
| hotpotqa | e3v | 11.3 | 12.43 | +1.13 [+0.94, +1.32] | ROOM | R.mention +1.41 | 73.85 | +0.70 [+0.56, +0.86] |
| 2wiki | e3v | 9.55 | 10.84 | +1.30 [+1.09, +1.50] | ROOM | S.hyperlink +1.81 | 80.05 | +0.75 [+0.59, +0.91] |

**Verdict (8.11, e3v):** ROOM on multihoprag, squad, musique, hotpotqa, 2wiki: the next hub step is declared on those labels
