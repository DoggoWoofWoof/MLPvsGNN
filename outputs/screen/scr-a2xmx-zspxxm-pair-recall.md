A2xmx (docs/A2XMX_ZTOP50_ADOPTED.md): zspxxm (zspx with zret's reference cut to the pool's top 50 retrieved rows by rrf (ztop50's form), before its three increments) against zspx.

# scr-a2xmx-zspxxm-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of zspxxm's p@swa minus zspx's p@swa of the same split (zspx's card fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/A2XMX_ZTOP50_ADOPTED.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zspx R@5 | zspxxm R@5 (A2xmx's top-50 reference) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8030 | 0.8004 | -0.0026 [-0.0036, -0.0016] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9092 | 0.9116 | +0.0024 [+0.0003, +0.0045] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5073 | 0.4949 | -0.0124 [-0.0199, -0.0054] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9074 | 0.9080 | +0.0006 [-0.0017, +0.0029] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9351 | 0.9343 | -0.0008 [-0.0022, +0.0007] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3204 | 0.3291 | +0.0088 [-0.0006, +0.0183] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8039 | 0.8019 | -0.0020 [-0.0031, -0.0011] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9129 | 0.9127 | -0.0003 [-0.0023, +0.0018] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5901 | 0.5863 | -0.0038 [-0.0089, +0.0014] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8547 | 0.8515 | -0.0032 [-0.0061, +0.0000] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9312 | 0.9283 | -0.0030 [-0.0048, -0.0013] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2914 | 0.2660 | -0.0254 [-0.0353, -0.0159] | +0.0102, +0.0129 | 0.0233 | LOSS | LOSS |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
