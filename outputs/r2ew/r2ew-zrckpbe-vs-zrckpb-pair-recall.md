R2EW (docs/R2EW_R2E_UNDER_WQ.md): zrckpbe against zrckpb under WQ on the T4, beside, deciding nothing.

# r2ew-zrckpbe-vs-zrckpb-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| WQ-musique | metaqa | zero-shot | -0.0090 [-0.0150, -0.0026] | -0.0204, -0.0389 | 0.0621 | LOSS | **WITHIN** |
| WQ-musique | squad | in-domain | +0.0000 [-0.0022, +0.0022] | -0.0014, -0.0002 | 0.0075 | WITHIN | WITHIN |
| WQ-musique | musique | zero-shot | -0.0088 [-0.0153, -0.0018] | -0.0335, -0.0219 | 0.0566 | LOSS | **WITHIN** |
| WQ-musique | hotpotqa | in-domain | +0.0040 [+0.0015, +0.0066] | -0.0053, -0.0031 | 0.0087 | WITHIN | WITHIN |
| WQ-musique | 2wiki | in-domain | +0.0049 [+0.0032, +0.0065] | -0.0035, -0.0056 | 0.0093 | WITHIN | WITHIN |
| WQ-musique | webqsp | in-domain | -0.0071 [-0.0175, +0.0024] | -0.0012, +0.0023 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa | metaqa | zero-shot | -0.0056 [-0.0114, +0.0002] | +0.0267, +0.0220 | 0.0489 | WITHIN | WITHIN |
| WQ-hotpotqa | squad | in-domain | +0.0015 [-0.0008, +0.0038] | -0.0003, -0.0001 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa | musique | in-domain | -0.0050 [-0.0115, +0.0014] | +0.0024, +0.0043 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa | hotpotqa | zero-shot | +0.0033 [-0.0003, +0.0066] | +0.0022, +0.0059 | 0.0089 | WITHIN | WITHIN |
| WQ-hotpotqa | 2wiki | in-domain | +0.0003 [-0.0015, +0.0021] | -0.0015, +0.0013 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa | webqsp | in-domain | -0.0063 [-0.0173, +0.0050] | +0.0035, +0.0011 | 0.0075 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: WQ-musique NO_GAIN -> NO_GAIN; WQ-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: WQ-musique metaqa LOSS -> WITHIN; WQ-musique musique LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
