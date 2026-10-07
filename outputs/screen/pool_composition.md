# Pool composition (a diagnosis)

Per question, then averaged over questions: rows in the pool, the share of rows that retrieval ranked (rrf > 0), golds in the pool, and the share of those golds that retrieval ranked.

| carve | questions | rows (median) | ranked rows (median) | ranked share | golds in pool | golds ranked share |
|---|---:|---:|---:|---:|---:|---:|
| metaqa=fit | 5960 | 2017 (2037) | 205.4 (206) | 0.102 | 6.36 | 0.427 |
| squad=fit | 5856 | 50 (50) | 50.0 (50) | 1.000 | 0.98 | 1.000 |
| hotpotqa=fit | 5930 | 94 (90) | 63.2 (60) | 0.693 | 1.96 | 0.946 |
| 2wiki=fit | 5928 | 106 (104) | 57.2 (55) | 0.557 | 2.29 | 0.802 |
| musique=s1fit | 4601 | 2092 (2134) | 369.0 (357) | 0.178 | 2.13 | 0.959 |
| metaqa=s1eval | 9785 | 2018 (2037) | 204.8 (206) | 0.102 | 6.41 | 0.422 |
| squad=s1eval | 11873 | 50 (50) | 50.0 (50) | 1.000 | 0.98 | 1.000 |
| musique=s1eval | 2417 | 2092 (2132) | 378.5 (365) | 0.182 | 2.40 | 0.931 |
| hotpotqa=s1eval | 7405 | 94 (90) | 63.0 (60) | 0.692 | 1.96 | 0.944 |
| 2wiki=s1eval | 12576 | 106 (104) | 58.0 (56) | 0.567 | 2.34 | 0.834 |
| webqsp=s1eval | 1503 | 2120 (2124) | 322.0 (308) | 0.152 | 3.32 | 0.581 |
