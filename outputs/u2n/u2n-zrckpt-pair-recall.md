U2n (docs/U2N_TYPED_CHANNELS.md): zrckpt against KG's zrckp on the T4, deciding.

# u2n-zrckpt-pair re-called under the seed null: **PROMISING** (filed: MIXED)

R@5 of the screen's p@swa minus step 1's p@swa of the same split, on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---|---|---:|---|---|
| L-musique | metaqa | in-domain | -0.0002 [-0.0011, +0.0006] | +0.0041, +0.0043 | 0.0084 | WITHIN | WITHIN |
| L-musique | squad | in-domain | +0.0000 [-0.0018, +0.0017] | -0.0005, -0.0009 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | +0.0098 [+0.0055, +0.0144] | -0.0015, -0.0012 | 0.0075 | GAIN | GAIN |
| L-musique | hotpotqa | in-domain | +0.0026 [+0.0005, +0.0046] | +0.0031, +0.0014 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | +0.0042 [+0.0026, +0.0056] | +0.0055, +0.0062 | 0.0117 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | -0.0053 [-0.0137, +0.0034] | -0.0047, -0.0121 | 0.0184 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | +0.0001 [-0.0008, +0.0010] | -0.0024, -0.0014 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | +0.0011 [-0.0008, +0.0030] | -0.0009, +0.0010 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | +0.0019 [-0.0031, +0.0069] | +0.0085, -0.0030 | 0.0127 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | +0.0080 [+0.0051, +0.0108] | +0.0058, +0.0012 | 0.0084 | GAIN | **WITHIN** |
| L-hotpotqa | 2wiki | in-domain | -0.0005 [-0.0019, +0.0011] | -0.0002, -0.0000 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | -0.0108 [-0.0221, -0.0006] | +0.0126, +0.0110 | 0.0237 | LOSS | **WITHIN** |

Re-call: reads with a GAIN: 1 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique PROMISING -> PROMISING; L-hotpotqa MIXED -> NO_GAIN.
Changed calls: L-hotpotqa hotpotqa GAIN -> WITHIN; L-hotpotqa webqsp LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
