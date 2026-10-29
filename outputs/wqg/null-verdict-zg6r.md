# WQG: zg6r's seed null under WQ: **FLOORS_READY**

Declared in docs/WQG_ZG6_NULL_UNDER_WQ.md. the arm is called under outputs/wqg/floors-<arm> from here: a later form's declaration cites it in place of zspk's stand-in; R2EG's and G6RW's filed verdicts stay as filed (their re-calls under the real floors are beside, and a flipped reading names a declared amendment, not an edit)

## zg6r's floors beside zspk's

floor = max(0.0075, 2 x the root mean square of the two seed-only differences) per read; zspk's are WQK's (the stand-in G6, G6RW and R2EG used).

| split | dataset | read | seed 0 R@5 | D1 | D2 | floor | zspk floor | |
|---|---|---|---:|---:|---:|---:|---:|---|
| WQ-musique | metaqa | zero-shot | 0.4105 | +0.0139 | -0.0020 | 0.0199 | 0.0439 | narrower |
| WQ-musique | squad | in-domain | 0.9105 | +0.0003 | +0.0023 | 0.0075 | 0.0075 | equal |
| WQ-musique | musique | zero-shot | 0.4106 | +0.0502 | +0.0512 | 0.1014 | 0.0545 | wider |
| WQ-musique | hotpotqa | in-domain | 0.9134 | -0.0024 | -0.0016 | 0.0075 | 0.0075 | equal |
| WQ-musique | 2wiki | in-domain | 0.9369 | +0.0038 | +0.0014 | 0.0075 | 0.0075 | equal |
| WQ-musique | webqsp | in-domain | 0.6501 | -0.0074 | -0.0026 | 0.0111 | 0.0174 | narrower |
| WQ-hotpotqa | metaqa | zero-shot | 0.3834 | +0.0521 | +0.0322 | 0.0866 | 0.0308 | wider |
| WQ-hotpotqa | squad | in-domain | 0.9111 | +0.0019 | +0.0013 | 0.0075 | 0.0075 | equal |
| WQ-hotpotqa | musique | in-domain | 0.5941 | -0.0021 | +0.0037 | 0.0075 | 0.0090 | narrower |
| WQ-hotpotqa | hotpotqa | zero-shot | 0.8529 | -0.0041 | -0.0044 | 0.0085 | 0.0075 | wider |
| WQ-hotpotqa | 2wiki | in-domain | 0.9420 | -0.0002 | +0.0012 | 0.0075 | 0.0075 | equal |
| WQ-hotpotqa | webqsp | in-domain | 0.6536 | -0.0053 | -0.0135 | 0.0205 | 0.0089 | wider |

Reads where zg6r's floor is wider than zspk's: 4 of 12 (WQ-musique musique, WQ-hotpotqa metaqa, WQ-hotpotqa hotpotqa, WQ-hotpotqa webqsp).
Narrower: 3 (WQ-musique metaqa, WQ-musique webqsp, WQ-hotpotqa musique).

## G6RW's zg6r against WQK's seed-0 zspk (the deciding comparison there), beside

| split | dataset | delta R@5 [95% CI] | filed | zspk floor | call | real floor | call |
|---|---|---|---|---:|---|---:|---|
| WQ-musique | metaqa | +0.0121 [+0.0058, +0.0184] | GAIN | 0.0439 | WITHIN | 0.0341 | WITHIN |
| WQ-musique | squad | -0.0005 [-0.0025, +0.0014] | WITHIN | 0.0075 | WITHIN | 0.0075 | WITHIN |
| WQ-musique | musique | -0.0484 [-0.0562, -0.0410] | LOSS | 0.0545 | WITHIN | 0.0814 | WITHIN |
| WQ-musique | hotpotqa | +0.0076 [+0.0053, +0.0097] | GAIN | 0.0075 | GAIN | 0.0075 | GAIN |
| WQ-musique | 2wiki | +0.0024 [+0.0009, +0.0038] | WITHIN | 0.0075 | WITHIN | 0.0075 | WITHIN |
| WQ-musique | webqsp | +0.0167 [+0.0039, +0.0288] | GAIN | 0.0174 | WITHIN | 0.0146 | **GAIN** |
| WQ-hotpotqa | metaqa | +0.0450 [+0.0380, +0.0519] | GAIN | 0.0308 | GAIN | 0.0650 | **WITHIN** |
| WQ-hotpotqa | squad | -0.0003 [-0.0022, +0.0015] | WITHIN | 0.0075 | WITHIN | 0.0075 | WITHIN |
| WQ-hotpotqa | musique | +0.0147 [+0.0086, +0.0207] | GAIN | 0.0090 | GAIN | 0.0076 | GAIN |
| WQ-hotpotqa | hotpotqa | +0.0033 [-0.0001, +0.0068] | WITHIN | 0.0075 | WITHIN | 0.0075 | WITHIN |
| WQ-hotpotqa | 2wiki | +0.0090 [+0.0074, +0.0107] | GAIN | 0.0075 | GAIN | 0.0075 | GAIN |
| WQ-hotpotqa | webqsp | +0.0383 [+0.0259, +0.0508] | GAIN | 0.0089 | GAIN | 0.0158 | GAIN |

Under zspk's floors: 5 GAIN, 0 LOSS, reads ADOPT_FOR_FULL_RUN. Under the real floors (zg6r and zspk): 5 GAIN, 0 LOSS, reads ADOPT_FOR_FULL_RUN. Changed calls: WQ-musique webqsp WITHIN -> GAIN; WQ-hotpotqa metaqa GAIN -> WITHIN.
The form's filed re-call (outputs/g6rw/g6rw-zg6r-pair-recall.json) agrees with the zspk-floor calls.

## G6RW's zg6r against WQK's seed-0 zg6 (beside there), beside

| split | dataset | delta R@5 [95% CI] | filed | zspk floor | call | real floor | call |
|---|---|---|---|---:|---|---:|---|
| WQ-musique | metaqa | -0.0288 [-0.0345, -0.0230] | LOSS | 0.0439 | WITHIN | 0.0476 | WITHIN |
| WQ-musique | squad | +0.0008 [-0.0013, +0.0028] | WITHIN | 0.0075 | WITHIN | 0.0075 | WITHIN |
| WQ-musique | musique | -0.0912 [-0.1007, -0.0819] | LOSS | 0.0545 | LOSS | 0.0738 | LOSS |
| WQ-musique | hotpotqa | +0.0024 [+0.0003, +0.0045] | WITHIN | 0.0075 | WITHIN | 0.0075 | WITHIN |
| WQ-musique | 2wiki | -0.0021 [-0.0037, -0.0005] | WITHIN | 0.0075 | WITHIN | 0.0075 | WITHIN |
| WQ-musique | webqsp | +0.0078 [-0.0051, +0.0209] | WITHIN | 0.0174 | WITHIN | 0.0122 | WITHIN |
| WQ-hotpotqa | metaqa | -0.0333 [-0.0393, -0.0275] | LOSS | 0.0308 | LOSS | 0.0614 | **WITHIN** |
| WQ-hotpotqa | squad | -0.0003 [-0.0024, +0.0016] | WITHIN | 0.0075 | WITHIN | 0.0075 | WITHIN |
| WQ-hotpotqa | musique | -0.0020 [-0.0075, +0.0034] | WITHIN | 0.0090 | WITHIN | 0.0080 | WITHIN |
| WQ-hotpotqa | hotpotqa | +0.0346 [+0.0309, +0.0386] | GAIN | 0.0075 | GAIN | 0.0099 | GAIN |
| WQ-hotpotqa | 2wiki | +0.0006 [-0.0010, +0.0023] | WITHIN | 0.0075 | WITHIN | 0.0075 | WITHIN |
| WQ-hotpotqa | webqsp | +0.0116 [-0.0010, +0.0242] | WITHIN | 0.0089 | WITHIN | 0.0199 | WITHIN |

Under zspk's floors: 1 GAIN, 2 LOSS, reads NOT_ADOPTED. Under the real floors (zg6r and zg6): 1 GAIN, 1 LOSS, reads NOT_ADOPTED. Changed calls: WQ-hotpotqa metaqa LOSS -> WITHIN.
The form's filed re-call (outputs/g6rw/g6rw-zg6r-vs-zg6-pair-recall.json) agrees with the zspk-floor calls.

Amendment: no reading flips: the filed verdicts stand.

## Seconds and memory per fit

| fit | train s | read s | train GPU GB | read GPU GB | peak RSS GB |
|---|---:|---:|---:|---:|---:|
| scr-wqg-zg6r-s1 | 1710.8697 | 1055.3963 | 2.0284 | 4.5913 | 4.5500 |
| scr-wqg-zg6r-s1-hp | 4051.5467 | 1086.0380 | 6.1021 | 6.1021 | 4.6700 |
| scr-wqg-zg6r-s2 | 1649.9860 | 903.7152 | 2.0424 | 4.5892 | 4.6000 |
| scr-wqg-zg6r-s2-hp | 3709.4754 | 940.1901 | 5.7511 | 5.7511 | 4.6800 |
| scr-g6rw-zg6r | 1818.9762 | 1088.1084 | 2.0508 | 4.5888 | 4.5600 |
| scr-g6rw-zg6r-hp | 3957.5031 | 961.7427 | 6.0079 | 6.0079 | 4.6700 |
