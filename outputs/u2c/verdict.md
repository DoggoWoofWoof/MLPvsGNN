# U2c: TARGET_MET on W2_S

Chosen: `g:f3+cut_cos_7000@0.9` (P1g's `f3+cut_cos_7000`, gated True, tau 0.9). Declared in docs/U2C_POOLS_ON_CHOSEN.md.

Headline graph: W2_S (holds hotpotqa and 2wiki at 0.95: {'hotpotqa': True, '2wiki': True}).

## today graph: TARGET_MET, `g:f2+cut_rrf_5000@0.95`, cost 1817.4 (today's pool 699.7)

| dataset | target | ALL | rows | rows p90 | today ALL | today rows |
|---|---:|---:|---:|---:|---:|---:|
| metaqa | 0.95 | 0.9679 | 3510.8 | 3629.0 | 0.9519 | 2021.7 |
| squad | 0.95 | 0.9793 | 50.0 | 50.0 | 0.9793 | 50.0 |
| musique | 0.95 | 0.9511 | 5617.0 | 6235.7 | 0.8846 | 2093.5 |
| hotpotqa | 0.95 | 0.9914 | 935.8 | 1402.3 | 0.9556 | 94.1 |
| 2wiki | 0.95 | 0.9699 | 1264.0 | 1632.0 | 0.9525 | 105.1 |
| webqsp (s1eval, beside) | - | 0.9082 | 4387.7 | 5954.0 | 0.841 | 2119.9 |

## chosen graph: TARGET_MET, `g:f3+cut_cos_7000@0.9`, cost 1953.5 (today's pool 716.5)

| dataset | target | ALL | rows | rows p90 | today ALL | today rows |
|---|---:|---:|---:|---:|---:|---:|
| metaqa | 0.95 | 0.9719 | 3929.0 | 4017.0 | 0.9519 | 2021.7 |
| squad | 0.95 | 0.9793 | 50.0 | 50.0 | 0.9793 | 50.0 |
| musique | 0.95 | 0.9511 | 5650.9 | 6584.4 | 0.8879 | 2067.2 |
| hotpotqa | 0.95 | 0.9542 | 963.8 | 1384.2 | 0.9138 | 119.6 |
| 2wiki | 0.95 | 0.9612 | 1407.3 | 1771.5 | 0.9325 | 162.2 |
| webqsp (s1eval, beside) | - | 0.8776 | 4555.1 | 6055.6 | 0.841 | 2119.9 |

## Ledger: the chosen pool's misses on the chosen graph

| dataset | golds | missed | TAU_CUT | BUDGET | UNREACHED | unreached listed | one link | KIND_T holds | today's graph reaches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| metaqa | 11367 | 284 | 187 | 19 | 78 | 4 | 0 | 0 | 0 |
| squad | 1498 | 31 | 0 | 21 | 10 | 9 | 0 | 9 | 4 |
| musique | 4064 | 82 | 29 | 7 | 46 | 0 | 0 | 0 | 2 |
| hotpotqa | 3016 | 72 | 1 | 22 | 49 | 12 | 0 | 28 | 42 |
| 2wiki | 3646 | 58 | 0 | 8 | 50 | 12 | 0 | 43 | 30 |
| webqsp | 10583 | 2376 | 813 | 114 | 1449 | 7 | 21 | 0 | 0 |

Beside: P1g2 on today's graph without a cut: TARGET_MET `g:f1.5+cut_cos_5000`, cost 1975.5.

