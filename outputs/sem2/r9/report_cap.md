# SEM2 R9 part A9: the cap ladder

Declared in docs/SEM2_EDGE_OPERATORS.md section 17. Dev slice (every eighth question of R8's population); budget 1500; rungs (hop-1 cap / out-neighbour cap) 100/50, 300/150, 1000/500.

| dataset | slice | scorer | rung | candidate recall % | completion % | proven share | recovered share % | median candidates |
|---|---:|---|---|---:|---:|---:|---:|---:|
| musique | 63 | W0 | 100/50 | 27.0 | 27.0 | 1.0000 | 29.58 | 3032 |
| musique | 63 | W0 | 300/150 | 55.6 | 27.0 | 1.0000 | 30.99 | 8193 |
| musique | 63 | W0 | 1000/500 | 90.5 | not read | 0.9683 | 28.99 | 21709 |
| musique | 63 | W1 | 100/50 | 34.9 | 33.3 | 1.0000 | 30.99 | 2863 |
| musique | 63 | W1 | 300/150 | 60.3 | 25.4 | 1.0000 | 23.94 | 7925 |
| musique | 63 | W1 | 1000/500 | 92.1 | not read | 0.8571 | 20.0 | 23834 |
| metaqa | 75 | W0 | 100/50 | 57.3 | 42.7 | 1.0000 | 48.61 | 2105 |
| metaqa | 75 | W0 | 300/150 | 96.0 | 25.3 | 1.0000 | 29.87 | 5893 |
| metaqa | 75 | W0 | 1000/500 | 100.0 | not read | 0.9867 | 29.37 | 10761 |
| metaqa | 75 | W1 | 100/50 | 49.3 | 21.3 | 1.0000 | 37.85 | 2268 |
| metaqa | 75 | W1 | 300/150 | 94.7 | 12.0 | 1.0000 | 14.47 | 6724 |
| metaqa | 75 | W1 | 1000/500 | 98.7 | 12.0 | 1.0000 | 13.91 | 11242 |
| webqsp | 75 | W0 | 100/50 | 42.7 | 36.0 | 1.0000 | 34.9 | 2132 |
| webqsp | 75 | W0 | 300/150 | 54.7 | 34.7 | 1.0000 | 24.05 | 8996 |
| webqsp | 75 | W0 | 1000/500 | 62.7 | not read | 0.9467 | 22.62 | 29383 |
| webqsp | 75 | W1 | 100/50 | 37.3 | 24.0 | 1.0000 | 20.82 | 2526 |
| webqsp | 75 | W1 | 300/150 | 46.7 | not read | 0.9733 | 15.43 | 11935 |
| webqsp | 75 | W1 | 1000/500 | 60.0 | not read | 0.8533 | 11.46 | 45094 |
| hotpotqa | 31 | W0 | 100/50 | 45.2 | 45.2 | 1.0000 | 45.16 | 2432 |
| hotpotqa | 31 | W0 | 300/150 | 90.3 | 41.9 | 1.0000 | 41.94 | 9902 |
| hotpotqa | 31 | W0 | 1000/500 | 100.0 | not read | 0.8065 | 12.0 | 46299 |
| hotpotqa | 31 | W1 | 100/50 | 41.9 | 41.9 | 1.0000 | 41.94 | 2322 |
| hotpotqa | 31 | W1 | 300/150 | 93.5 | not read | 0.9677 | 33.33 | 10863 |
| hotpotqa | 31 | W1 | 1000/500 | 96.8 | not read | 0.9032 | 17.86 | 54574 |

Hop-1 need of the missed golds (the largest rank, in the scorer's order over the hop-1 field, of a question's missed hop-1 golds):

| dataset | questions with a hop-1 missed gold | scorer | need <= 100 % | <= 300 % | <= 1000 % | median | p90 | median / field | p90 / field |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| musique | 46 | W0 | 26.09 | 50.0 | 89.13 | 310.5 | 1010.5 | 0.3176 | 0.7552 |
| musique | 46 | W1 | 36.96 | 58.7 | 93.48 | 174.0 | 890.0 | 0.1867 | 0.6509 |
| metaqa | 47 | W0 | 61.7 | 95.74 | 100.0 | 60.0 | 212.59999999999997 | 0.3211 | 0.8238 |
| metaqa | 47 | W1 | 51.06 | 97.87 | 100.0 | 97.0 | 247.2 | 0.5277 | 0.8751 |
| webqsp | 70 | W0 | 42.86 | 52.86 | 60.0 | 200.0 | 7337.200000000001 | 0.0266 | 0.887 |
| webqsp | 70 | W1 | 40.0 | 50.0 | 64.29 | 310.0 | 9088.4 | 0.0221 | 0.7633 |
| hotpotqa | 30 | W0 | 46.67 | 90.0 | 100.0 | 101.5 | 275.7000000000001 | 0.2079 | 0.5342 |
| hotpotqa | 30 | W1 | 43.33 | 96.67 | 100.0 | 150.5 | 255.90000000000003 | 0.2627 | 0.5718 |

Readings (decide nothing):

- W0: CAP_BINDS True, COMPLETION_FOLLOWS False; recall gain C3 - C1 {'musique': 63.5, 'metaqa': 42.67, 'webqsp': 20.0, 'hotpotqa': 54.84}; completion gain {'musique': None, 'metaqa': None, 'webqsp': None, 'hotpotqa': None}
- W1: CAP_BINDS True, COMPLETION_FOLLOWS False; recall gain C3 - C1 {'musique': 57.14, 'metaqa': 49.34, 'webqsp': 22.67, 'hotpotqa': 54.83}; completion gain {'musique': None, 'metaqa': -9.33, 'webqsp': None, 'hotpotqa': None}

Flat ceiling (best completion read over the rungs and scorers): {'musique': 33.33, 'metaqa': 42.67, 'webqsp': 36.0, 'hotpotqa': 45.16}
Missing: 2wiki.
