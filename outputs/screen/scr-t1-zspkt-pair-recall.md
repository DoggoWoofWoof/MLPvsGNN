T1 (docs/T1_GRANULARITY_TRANSFORMS.md): zspkt (zspk trained with metaqa's granularity transforms) against zspk.

# scr-t1-zspkt-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of zspkt's p@swa minus zspk's p@swa of the same split (zspk's card fits, metaqa as filed), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/T1_GRANULARITY_TRANSFORMS.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zspk R@5 | zspkt R@5 (T1's metaqa) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8009 | 0.7989 | -0.0020 [-0.0037, -0.0003] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9114 | 0.9096 | -0.0018 [-0.0037, +0.0001] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5012 | 0.4448 | -0.0564 [-0.0646, -0.0485] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9080 | 0.9063 | -0.0017 [-0.0037, +0.0004] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9383 | 0.9371 | -0.0012 [-0.0024, +0.0003] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3229 | 0.3226 | -0.0003 [-0.0143, +0.0138] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8040 | 0.7968 | -0.0072 [-0.0088, -0.0055] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9135 | 0.9136 | +0.0001 [-0.0019, +0.0020] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5754 | 0.5798 | +0.0044 [-0.0010, +0.0100] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8499 | 0.8473 | -0.0026 [-0.0056, +0.0006] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9333 | 0.9367 | +0.0034 [+0.0019, +0.0050] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2680 | 0.2741 | +0.0062 [-0.0077, +0.0209] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
