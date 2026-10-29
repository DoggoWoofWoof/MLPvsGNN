R2WS (docs/R2WS_R2_SEEDS_UNDER_WQ.md): zrckpb at seed 1 against zrckp at seed 1 under WQ on the T4, with the rule joining the seeds.

# r2ws-zrckpb-s1-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| WQ-musique | metaqa | zero-shot | +0.0266 [+0.0211, +0.0318] | -0.0204, -0.0389 | 0.0621 | GAIN | **WITHIN** |
| WQ-musique | squad | in-domain | -0.0004 [-0.0027, +0.0018] | -0.0014, -0.0002 | 0.0075 | WITHIN | WITHIN |
| WQ-musique | musique | zero-shot | +0.0373 [+0.0296, +0.0452] | -0.0335, -0.0219 | 0.0566 | GAIN | **WITHIN** |
| WQ-musique | hotpotqa | in-domain | +0.0015 [-0.0013, +0.0043] | -0.0053, -0.0031 | 0.0087 | WITHIN | WITHIN |
| WQ-musique | 2wiki | in-domain | +0.0036 [+0.0020, +0.0052] | -0.0035, -0.0056 | 0.0093 | WITHIN | WITHIN |
| WQ-musique | webqsp | in-domain | +0.0134 [+0.0018, +0.0247] | -0.0012, +0.0023 | 0.0075 | GAIN | GAIN |
| WQ-hotpotqa | metaqa | zero-shot | +0.0087 [+0.0024, +0.0149] | +0.0267, +0.0220 | 0.0489 | GAIN | **WITHIN** |
| WQ-hotpotqa | squad | in-domain | -0.0026 [-0.0048, -0.0004] | -0.0003, -0.0001 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa | musique | in-domain | -0.0175 [-0.0237, -0.0114] | +0.0024, +0.0043 | 0.0075 | LOSS | LOSS |
| WQ-hotpotqa | hotpotqa | zero-shot | -0.0125 [-0.0160, -0.0090] | +0.0022, +0.0059 | 0.0089 | LOSS | LOSS |
| WQ-hotpotqa | 2wiki | in-domain | +0.0022 [+0.0004, +0.0039] | -0.0015, +0.0013 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa | webqsp | in-domain | +0.0024 [-0.0094, +0.0134] | +0.0035, +0.0011 | 0.0075 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 1 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: WQ-musique PROMISING -> PROMISING; WQ-hotpotqa MIXED -> NO_GAIN.
Changed calls: WQ-musique metaqa GAIN -> WITHIN; WQ-musique musique GAIN -> WITHIN; WQ-hotpotqa metaqa GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
