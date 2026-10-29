R2GW (docs/R2GW_R2G_UNDER_WQ.md): zrckpb against zrckp under WQ on the T4, beside, deciding nothing.

# r2gw-zrckpb-pair re-called under the seed null: **PROMISING** (filed: PROMISING)

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| WQ-musique | metaqa | zero-shot | +0.0185 [+0.0121, +0.0249] | -0.0204, -0.0389 | 0.0621 | GAIN | **WITHIN** |
| WQ-musique | squad | in-domain | -0.0025 [-0.0046, -0.0004] | -0.0014, -0.0002 | 0.0075 | WITHIN | WITHIN |
| WQ-musique | musique | zero-shot | +0.0027 [-0.0044, +0.0093] | -0.0335, -0.0219 | 0.0566 | WITHIN | WITHIN |
| WQ-musique | hotpotqa | in-domain | -0.0057 [-0.0082, -0.0034] | -0.0053, -0.0031 | 0.0087 | WITHIN | WITHIN |
| WQ-musique | 2wiki | in-domain | -0.0042 [-0.0058, -0.0025] | -0.0035, -0.0056 | 0.0093 | WITHIN | WITHIN |
| WQ-musique | webqsp | in-domain | +0.0144 [+0.0034, +0.0257] | -0.0012, +0.0023 | 0.0075 | GAIN | GAIN |
| WQ-hotpotqa | metaqa | zero-shot | +0.0533 [+0.0475, +0.0590] | +0.0267, +0.0220 | 0.0489 | GAIN | GAIN |
| WQ-hotpotqa | squad | in-domain | -0.0028 [-0.0049, -0.0008] | -0.0003, -0.0001 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa | musique | in-domain | -0.0055 [-0.0119, +0.0011] | +0.0024, +0.0043 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa | hotpotqa | zero-shot | -0.0026 [-0.0059, +0.0008] | +0.0022, +0.0059 | 0.0089 | WITHIN | WITHIN |
| WQ-hotpotqa | 2wiki | in-domain | +0.0005 [-0.0011, +0.0022] | -0.0015, +0.0013 | 0.0075 | WITHIN | WITHIN |
| WQ-hotpotqa | webqsp | in-domain | +0.0106 [+0.0002, +0.0218] | +0.0035, +0.0011 | 0.0075 | GAIN | GAIN |

Re-call: reads with a GAIN: 3 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: WQ-musique PROMISING -> PROMISING; WQ-hotpotqa PROMISING -> PROMISING.
Changed calls: WQ-musique metaqa GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
