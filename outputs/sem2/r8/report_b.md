# SEM2 R8 part B: the router

Declared in docs/SEM2_EDGE_OPERATORS.md sections 16.6, 16.8 and 16.13. Completion rate in percent of the population (completable at 1,500), main budget 1,500.

| dataset | population | lambda | W0 qcos | W1 lcos | W2 plaus | W3 routed | W4 other q | W5 random | W6 | W7 | W3-W0 [95%] | call | W3-W4 | control |
|---|---:|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| musique | 503 | 0.0 | 30.0 | 31.6 | 21.3 | 21.3 | 21.3 | 9.7 | 30.6 | 21.7 | -8.75 [-13.52, -3.98] | BELOW | +0.00 | LEVEL |
| metaqa | 600 | 0.0 | 44.7 | 26.8 | 37.3 | 37.3 | 37.3 | 21.5 | 43.2 | 29.8 | -7.33 [-11.67, -3.17] | BELOW | +0.00 | LEVEL |
| webqsp | 600 | 0.0 | 32.2 | 22.5 | 26.3 | 26.3 | 26.3 | 1.5 | 36.7 | 32.3 | -5.83 [-10.00, -1.83] | BELOW | +0.00 | LEVEL |
| hotpotqa | 242 | 0.25 | 60.7 | 50.0 | 46.7 | 50.8 | 43.8 | 16.1 | 57.9 | 39.3 | -9.92 [-17.36, -2.48] | BELOW | +7.02 | ABOVE |
| 2wiki | 600 | 0.25 | 14.2 | 35.7 | 29.7 | 28.7 | 30.0 | 6.5 | 26.7 | 27.0 | +14.50 [+10.67, +18.33] | ABOVE | -1.33 | LEVEL |

Verdict: **NOT_CARRIED**. ROOM on: musique, metaqa, webqsp, hotpotqa, 2wiki. Missing: none.

Beside, at unit lambda (16.13; decides nothing): W6 - W0 and the control W6 - W7.

| dataset | W6 - W0 [95%] | call | W6 - W7 | control | W2 - W0 | call |
|---|---|---|---|---|---|---|
| musique | +0.60 [-3.38, +4.17] | LEVEL | +8.95 | ABOVE | -8.75 | BELOW |
| metaqa | -1.50 [-5.17, +2.00] | LEVEL | +13.33 | ABOVE | -7.33 | BELOW |
| webqsp | +4.50 [+0.83, +8.17] | ABOVE | +4.33 | ABOVE | -5.83 | BELOW |
| hotpotqa | -2.89 [-9.09, +3.31] | LEVEL | +18.60 | ABOVE | -14.05 | BELOW |
| 2wiki | +12.50 [+9.17, +15.83] | ABOVE | -0.33 | LEVEL | +15.50 | ABOVE |

Beside verdict at unit lambda: NOT_CARRIED.
