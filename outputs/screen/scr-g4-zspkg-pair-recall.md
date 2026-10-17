G4 (docs/G4_DEEPER_PROPAGATION.md): zspkg (zspk with its propagation trained end to end) against zspk.

# scr-g4-zspkg-pair re-called under the seed null: **PROMISING** (filed: PROMISING)

R@5 of zspkg's p@swa minus zspk's p@swa of the same split (zspk's card fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/G4_DEEPER_PROPAGATION.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zspk R@5 | zspkg R@5 (zspk with its propagation trained end to end) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8009 | 0.8015 | +0.0006 [-0.0004, +0.0015] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9114 | 0.9111 | -0.0003 [-0.0023, +0.0016] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5012 | 0.5000 | -0.0012 [-0.0066, +0.0043] | +0.0503, -0.0079 | 0.0720 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | 0.9080 | 0.9084 | +0.0003 [-0.0017, +0.0024] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9383 | 0.9355 | -0.0028 [-0.0042, -0.0015] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3229 | 0.3161 | -0.0067 [-0.0152, +0.0019] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8040 | 0.8044 | +0.0004 [-0.0005, +0.0012] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9135 | 0.9132 | -0.0003 [-0.0021, +0.0015] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5754 | 0.5847 | +0.0093 [+0.0042, +0.0141] | +0.0015, +0.0023 | 0.0075 | GAIN | GAIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8499 | 0.8523 | +0.0024 [-0.0003, +0.0051] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9333 | 0.9323 | -0.0010 [-0.0023, +0.0004] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2680 | 0.2832 | +0.0152 [+0.0054, +0.0246] | +0.0102, +0.0129 | 0.0233 | GAIN | **WITHIN** |

Re-call: reads with a GAIN: 1 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa PROMISING -> PROMISING.
Changed calls: L-hotpotqa webqsp GAIN -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
