# WQG: zg6's seed null under WQ: **FLOORS_READY**

Declared in docs/WQG_ZG6_NULL_UNDER_WQ.md. the arm is called under outputs/wqg/floors-<arm> from here: a later form's declaration cites it in place of zspk's stand-in; R2EG's and G6RW's filed verdicts stay as filed (their re-calls under the real floors are beside, and a flipped reading names a declared amendment, not an edit)

## zg6's floors beside zspk's

floor = max(0.0075, 2 x the root mean square of the two seed-only differences) per read; zspk's are WQK's (the stand-in G6, G6RW and R2EG used).

| split | dataset | read | seed 0 R@5 | D1 | D2 | floor | zspk floor | |
|---|---|---|---:|---:|---:|---:|---:|---|
| WQ-musique | metaqa | zero-shot | 0.4393 | -0.0154 | -0.0428 | 0.0643 | 0.0439 | wider |
| WQ-musique | squad | in-domain | 0.9097 | +0.0016 | +0.0027 | 0.0075 | 0.0075 | equal |
| WQ-musique | musique | zero-shot | 0.5018 | +0.0150 | +0.0093 | 0.0250 | 0.0545 | narrower |
| WQ-musique | hotpotqa | in-domain | 0.9109 | +0.0015 | +0.0016 | 0.0075 | 0.0075 | equal |
| WQ-musique | 2wiki | in-domain | 0.9390 | +0.0012 | +0.0024 | 0.0075 | 0.0075 | equal |
| WQ-musique | webqsp | in-domain | 0.6423 | -0.0012 | -0.0093 | 0.0133 | 0.0174 | narrower |
| WQ-hotpotqa | metaqa | zero-shot | 0.4167 | -0.0037 | +0.0025 | 0.0075 | 0.0308 | narrower |
| WQ-hotpotqa | squad | in-domain | 0.9114 | -0.0004 | +0.0003 | 0.0075 | 0.0075 | equal |
| WQ-hotpotqa | musique | in-domain | 0.5961 | -0.0065 | -0.0017 | 0.0095 | 0.0090 | wider |
| WQ-hotpotqa | hotpotqa | zero-shot | 0.8183 | +0.0078 | +0.0012 | 0.0112 | 0.0075 | wider |
| WQ-hotpotqa | 2wiki | in-domain | 0.9414 | -0.0017 | +0.0007 | 0.0075 | 0.0075 | equal |
| WQ-hotpotqa | webqsp | in-domain | 0.6421 | -0.0081 | -0.0110 | 0.0193 | 0.0089 | wider |

Reads where zg6's floor is wider than zspk's: 4 of 12 (WQ-musique metaqa, WQ-hotpotqa musique, WQ-hotpotqa hotpotqa, WQ-hotpotqa webqsp).
Narrower: 3 (WQ-musique musique, WQ-musique webqsp, WQ-hotpotqa metaqa).

## R2EG's zg6e against WQK's seed-0 zg6, beside

| split | dataset | delta R@5 [95% CI] | filed | zspk floor | call | real floor | call |
|---|---|---|---|---:|---|---:|---|
| WQ-musique | metaqa | -0.0443 [-0.0505, -0.0383] | LOSS | 0.0439 | LOSS | 0.0643 | **WITHIN** |
| WQ-musique | squad | +0.0004 [-0.0018, +0.0026] | WITHIN | 0.0075 | WITHIN | 0.0075 | WITHIN |
| WQ-musique | musique | -0.0180 [-0.0250, -0.0104] | LOSS | 0.0545 | WITHIN | 0.0250 | WITHIN |
| WQ-musique | hotpotqa | +0.0036 [+0.0013, +0.0058] | WITHIN | 0.0075 | WITHIN | 0.0075 | WITHIN |
| WQ-musique | 2wiki | -0.0010 [-0.0027, +0.0006] | WITHIN | 0.0075 | WITHIN | 0.0075 | WITHIN |
| WQ-musique | webqsp | +0.0015 [-0.0106, +0.0149] | WITHIN | 0.0174 | WITHIN | 0.0133 | WITHIN |
| WQ-hotpotqa | metaqa | -0.0320 [-0.0374, -0.0266] | LOSS | 0.0308 | LOSS | 0.0075 | LOSS |
| WQ-hotpotqa | squad | -0.0006 [-0.0029, +0.0018] | WITHIN | 0.0075 | WITHIN | 0.0075 | WITHIN |
| WQ-hotpotqa | musique | -0.0033 [-0.0092, +0.0024] | WITHIN | 0.0090 | WITHIN | 0.0095 | WITHIN |
| WQ-hotpotqa | hotpotqa | +0.0097 [+0.0059, +0.0134] | GAIN | 0.0075 | GAIN | 0.0112 | **WITHIN** |
| WQ-hotpotqa | 2wiki | -0.0015 [-0.0032, +0.0002] | WITHIN | 0.0075 | WITHIN | 0.0075 | WITHIN |
| WQ-hotpotqa | webqsp | -0.0078 [-0.0203, +0.0043] | WITHIN | 0.0089 | WITHIN | 0.0193 | WITHIN |

Under zspk's floors: 1 GAIN, 2 LOSS, reads NOT_ADOPTED. Under the real floors (zg6): 0 GAIN, 1 LOSS, reads NOT_ADOPTED. Changed calls: WQ-musique metaqa LOSS -> WITHIN; WQ-hotpotqa hotpotqa GAIN -> WITHIN.
The form's filed re-call (outputs/r2eg/r2eg-zg6e-pair-recall.json) agrees with the zspk-floor calls.

Amendment: no reading flips: the filed verdicts stand.

## Seconds and memory per fit

| fit | train s | read s | train GPU GB | read GPU GB | peak RSS GB |
|---|---:|---:|---:|---:|---:|
| scr-wqg-zg6-s1 | 1312.5827 | 700.3186 | 2.2346 | 5.2347 | 4.5200 |
| scr-wqg-zg6-s1-hp | 3214.3097 | 699.0747 | 9.3745 | 9.3745 | 4.5900 |
| scr-wqg-zg6-s2 | 1317.4025 | 683.3903 | 2.2864 | 5.2340 | 4.5800 |
| scr-wqg-zg6-s2-hp | 3241.5459 | 693.8638 | 8.5298 | 8.5298 | 4.5600 |
| scr-wqk-zg6-s0 | 1364.4681 | 706.3022 | 2.2806 | 5.2366 | 4.5400 |
| scr-wqk-zg6-s0-hp | 3355.5602 | 715.0131 | 9.1280 | 9.1280 | 4.5600 |
