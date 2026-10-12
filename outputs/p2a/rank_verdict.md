# P2a part 2: what today's lean fits lose with one link family masked (s1eval)

Each cell: the fit's R@5 unmasked minus masked (what it loses without the family) with its paired bootstrap 95% interval, then the label over R@5, FC@5 and hit@1 (a difference counts above 0.002 with its interval clear of 0). zrc is the MLP, zsp the GNN; webqsp is zero-shot.

## zrc

| dataset | unmasked R@5 | -S | -N | -K | -SNK (beside) |
| --- | ---: | --- | --- | --- | --- |
| metaqa | 0.7796 | +0.7716 [+0.7646, +0.7785] RELIES | +0.0000 [-0.0001, +0.0002] NOTHING | -0.0003 [-0.0009, +0.0003] RELIES | +0.7703 [+0.7631, +0.7772] RELIES |
| squad | 0.9107 | -0.0015 [-0.0037, +0.0005] NOTHING | +0.0007 [-0.0006, +0.0019] NOTHING | +0.0003 [-0.0009, +0.0016] NOTHING | -0.0019 [-0.0040, +0.0002] NOTHING |
| musique | 0.5669 | +0.0288 [+0.0213, +0.0358] RELIES | +0.0088 [+0.0037, +0.0135] RELIES | -0.0013 [-0.0050, +0.0023] NOTHING | +0.0534 [+0.0452, +0.0614] RELIES |
| hotpotqa | 0.9049 | +0.1686 [+0.1628, +0.1745] RELIES | +0.0001 [-0.0014, +0.0016] RELIES | -0.0001 [-0.0015, +0.0014] RELIES | +0.1904 [+0.1844, +0.1964] RELIES |
| 2wiki | 0.8697 | +0.1693 [+0.1651, +0.1737] MIXED | +0.0002 [-0.0010, +0.0013] RELIES | +0.0004 [-0.0007, +0.0015] RELIES | +0.2244 [+0.2196, +0.2289] MIXED |
| webqsp | 0.3071 | +0.2230 [+0.1991, +0.2467] RELIES | -0.0005 [-0.0043, +0.0032] NOTHING | -0.0004 [-0.0073, +0.0066] NOTHING | +0.2233 [+0.1999, +0.2473] RELIES |

## zsp

| dataset | unmasked R@5 | -S | -N | -K | -SNK (beside) |
| --- | ---: | --- | --- | --- | --- |
| metaqa | 0.7794 | +0.7732 [+0.7662, +0.7801] RELIES | -0.0001 [-0.0003, +0.0002] NOTHING | +0.0005 [-0.0002, +0.0013] RELIES | +0.7733 [+0.7660, +0.7802] RELIES |
| squad | 0.9099 | -0.0018 [-0.0040, +0.0004] NOTHING | +0.0009 [-0.0004, +0.0024] NOTHING | -0.0008 [-0.0024, +0.0008] RELIES | -0.0025 [-0.0050, -0.0003] BETTER_WITHOUT |
| musique | 0.5657 | +0.0255 [+0.0175, +0.0331] RELIES | +0.0137 [+0.0090, +0.0186] RELIES | +0.0038 [-0.0010, +0.0088] RELIES | +0.0528 [+0.0445, +0.0607] RELIES |
| hotpotqa | 0.9051 | +0.1679 [+0.1624, +0.1738] RELIES | +0.0003 [-0.0012, +0.0018] NOTHING | +0.0013 [-0.0003, +0.0029] RELIES | +0.1998 [+0.1940, +0.2060] RELIES |
| 2wiki | 0.8726 | +0.1653 [+0.1605, +0.1700] MIXED | -0.0001 [-0.0015, +0.0012] NOTHING | +0.0014 [+0.0001, +0.0028] RELIES | +0.2247 [+0.2198, +0.2295] MIXED |
| webqsp | 0.2677 | +0.1873 [+0.1651, +0.2106] RELIES | +0.0035 [-0.0007, +0.0076] NOTHING | +0.0130 [+0.0047, +0.0215] RELIES | +0.1899 [+0.1672, +0.2137] RELIES |

## Families

- **structural:** feature drop candidate: False; pool drop on every regime dataset (part 1): False; graph drop candidate: False.
- **ner:** feature drop candidate: False; pool drop on every regime dataset (part 1): False; graph drop candidate: False.
- **knn:** feature drop candidate: False; pool drop on every regime dataset (part 1): False; graph drop candidate: False.

## Beside

| dataset | zrc R@5 (filed) | zsp R@5 (filed) | edges per question S / N / K | rows bit-equal to step 1 |
| --- | --- | --- | --- | --- |
| metaqa | 0.7796 (0.7796) | 0.7794 (0.7794) | 9174.2 / 272.7 / 1328.5 | 9065/9785 |
| squad | 0.9107 (0.9107) | 0.9099 (0.9099) | 258.6 / 106.5 / 82.6 | 11861/11873 |
| musique | 0.5669 (0.5669) | 0.5657 (0.5657) | 40993.0 / 14900.4 / 2880.6 | 2136/2417 |
| hotpotqa | 0.9049 (0.9049) | 0.9051 (0.9051) | 258.1 / 162.7 / 87.6 | 7380/7405 |
| 2wiki | 0.8697 (0.8697) | 0.8726 (0.8726) | 267.2 / 158.3 / 67.6 | 12524/12576 |
| webqsp | 0.3071 (0.3071) | 0.2677 (0.2677) | 9996.2 / 3406.9 / 3640.7 | 1416/1503 |
