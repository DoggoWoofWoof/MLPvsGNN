# A5cp's screen verdict: zrckpc **NOT_ADOPTED**, zspkc **ADOPT_FOR_FULL_RUN**

Rule: per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A5cp's re-call of the arm, or not twelve reads (docs/A5CP_P1GB_POOLS.md, section 4; A5c's rule).

- **zrckpc** (zrckp with zck's coupled kind offsets on typed graphs (zret's output layer and base weight, trained jointly)) against zrckp's card fit of each split on P1g-b's pools (P1g-b's scr-p1gb-zrckp, scr-p1gb-zrckp-hp), after the seed-null re-call: 2 GAIN (L-musique webqsp, L-hotpotqa webqsp), 1 LOSS (L-hotpotqa 2wiki). Beside: L-musique metaqa WITHIN (+0.0004), L-musique webqsp GAIN (+0.0160), L-hotpotqa metaqa WITHIN (+0.0022), L-hotpotqa webqsp GAIN (+0.0366). Next: A5cb on P1g-b's pools: the coupled offsets bounded (each offset 0.2 tanh of its parameter), its own declared file.
- **zspkc** (zspk with zck's coupled kind offsets on typed graphs (zret's output layer and base weight, trained jointly), before its propagation) against zspk's card fit of each split on P1g-b's pools (P1g-b's scr-p1gb-zspk, scr-p1gb-zspk-hp), after the seed-null re-call: 3 GAIN (L-musique webqsp, L-hotpotqa musique, L-hotpotqa webqsp), 0 LOSS (none). Beside: L-musique metaqa WITHIN (-0.0023), L-musique webqsp GAIN (+0.0119), L-hotpotqa metaqa WITHIN (-0.0007), L-hotpotqa webqsp GAIN (+0.0455). Next: A5c's coupled offsets join the full run's models when it runs on P1g-b's pools: one file, declared before its numbers, after P1g-c's, A5cp's and A7p's verdicts, with the composition P1g-c's section 5 sets.

F1: only zspkc adopted: the full run's file names the offsets as a difference between its two models (F1).

zrckpc, each read beside, deciding nothing (FC@5 and hit@1 from `scr-a5cp-zrckpc` and its -hp; A5c's delta and call on P1c2's pools from `scr-a5c-zrckpc-pair-recall.json`):

| split | dataset | read | delta R@5 [95% CI] | call | FC@5 | hit@1 | A5c on P1c2's pools |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| L-musique | metaqa | in-domain | +0.0004 [-0.0006, +0.0015] | WITHIN | +0.0006 | +0.0027 | -0.0025 WITHIN |
| L-musique | squad | in-domain | -0.0009 [-0.0027, +0.0009] | WITHIN | -0.0009 | +0.0013 | -0.0018 WITHIN |
| L-musique | musique | zero-shot | +0.0098 [+0.0051, +0.0147] | WITHIN (filed GAIN) | +0.0108 | -0.0070 | +0.0084 WITHIN |
| L-musique | hotpotqa | in-domain | +0.0026 [+0.0003, +0.0050] | WITHIN | +0.0047 | +0.0088 | -0.0001 WITHIN |
| L-musique | 2wiki | in-domain | -0.0045 [-0.0060, -0.0030] | WITHIN | -0.0134 | +0.0007 | +0.0001 WITHIN |
| L-musique | webqsp | zero-shot | +0.0160 [+0.0039, +0.0280] | GAIN | +0.0086 | +0.0040 | +0.0189 GAIN |
| L-hotpotqa | metaqa | in-domain | +0.0022 [+0.0012, +0.0034] | WITHIN | +0.0018 | +0.0105 | -0.0005 WITHIN |
| L-hotpotqa | squad | in-domain | +0.0000 [-0.0019, +0.0019] | WITHIN | +0.0000 | +0.0008 | -0.0006 WITHIN |
| L-hotpotqa | musique | in-domain | +0.0000 [-0.0053, +0.0050] | WITHIN | -0.0054 | -0.0062 | -0.0018 WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0019 [-0.0051, +0.0011] | WITHIN | -0.0012 | -0.0117 | -0.0036 WITHIN |
| L-hotpotqa | 2wiki | in-domain | -0.0084 [-0.0100, -0.0066] | LOSS | -0.0249 | -0.0015 | -0.0018 WITHIN |
| L-hotpotqa | webqsp | zero-shot | +0.0366 [+0.0216, +0.0522] | GAIN | +0.0299 | +0.0373 | +0.0239 GAIN |

zrckpc's training forwards (typed / untyped, stamped by each fit): L-musique 5,918 / 17,759; L-hotpotqa 5,591 / 16,771.

zspkc, each read beside, deciding nothing (FC@5 and hit@1 from `scr-a5cp-zspkc` and its -hp; A5c's delta and call on P1c2's pools from `scr-a5c-zspkc-pair-recall.json`):

| split | dataset | read | delta R@5 [95% CI] | call | FC@5 | hit@1 | A5c on P1c2's pools |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| L-musique | metaqa | in-domain | -0.0023 [-0.0035, -0.0011] | WITHIN | -0.0009 | -0.0048 | +0.0021 WITHIN |
| L-musique | squad | in-domain | +0.0003 [-0.0013, +0.0019] | WITHIN | +0.0003 | -0.0019 | -0.0012 WITHIN |
| L-musique | musique | zero-shot | +0.0132 [+0.0079, +0.0183] | WITHIN (filed GAIN) | +0.0157 | +0.0017 | +0.0224 WITHIN |
| L-musique | hotpotqa | in-domain | +0.0004 [-0.0018, +0.0025] | WITHIN | +0.0018 | +0.0115 | -0.0016 WITHIN |
| L-musique | 2wiki | in-domain | +0.0021 [+0.0008, +0.0036] | WITHIN | +0.0069 | +0.0002 | -0.0014 WITHIN |
| L-musique | webqsp | zero-shot | +0.0119 [+0.0000, +0.0244] | GAIN | +0.0133 | +0.0080 | -0.0052 WITHIN |
| L-hotpotqa | metaqa | in-domain | -0.0007 [-0.0018, +0.0004] | WITHIN | +0.0007 | -0.0008 | +0.0006 WITHIN |
| L-hotpotqa | squad | in-domain | +0.0001 [-0.0019, +0.0020] | WITHIN | +0.0001 | -0.0012 | -0.0014 WITHIN |
| L-hotpotqa | musique | in-domain | +0.0122 [+0.0067, +0.0180] | GAIN | +0.0091 | -0.0029 | +0.0083 GAIN |
| L-hotpotqa | hotpotqa | zero-shot | +0.0045 [+0.0018, +0.0072] | WITHIN | +0.0101 | -0.0090 | -0.0031 WITHIN |
| L-hotpotqa | 2wiki | in-domain | +0.0044 [+0.0029, +0.0060] | WITHIN | +0.0119 | -0.0054 | +0.0041 WITHIN |
| L-hotpotqa | webqsp | zero-shot | +0.0455 [+0.0314, +0.0599] | GAIN | +0.0419 | +0.0373 | +0.0520 GAIN |

zspkc's training forwards (typed / untyped, stamped by each fit): L-musique 5,918 / 17,759; L-hotpotqa 5,591 / 16,771.

Beside, deciding nothing: musique in domain (L-hotpotqa musique, Claim 1's read): zrckpc WITHIN (+0.0000), zspkc GAIN (+0.0122).

This verdict closes only this form (A5c's coupled offsets on P1g-b's pools); docs/SCREENS.md, 3b. Which pools the full run uses is P1g-c's call (its section 5): on P1g-b's pools this verdict decides whether the offsets join, per arm; on P1c2's pools A5c's own verdict decides, and this one is reported beside.
