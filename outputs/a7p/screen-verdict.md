# A7p's screen verdict: zrckps **ADOPT_FOR_FULL_RUN**, zspks **ADOPT_FOR_FULL_RUN**

Rule: per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A7p's re-call of the arm, or not twelve reads (docs/A7P_P1GB_POOLS.md, section 4; A7's rule).

- **zrckps** (zrckp plus the A7 head) against zrckp's card fit of each split on P1g-b's pools (P1g-b's scr-p1gb-zrckp, scr-p1gb-zrckp-hp), after the seed-null re-call: 2 GAIN (L-musique webqsp, L-hotpotqa webqsp), 0 LOSS (none). Beside: L-musique metaqa WITHIN (+0.0022), L-musique webqsp GAIN (+0.0194), L-hotpotqa metaqa WITHIN (+0.0044), L-hotpotqa webqsp GAIN (+0.0273). Next: A7's columns join the full run's models when it runs on P1g-b's pools: one file, declared before its numbers, after P1g-c's, A5cp's, A7p's and A3vp's verdicts, with the composition P1g-c's section 5 sets; each of its splits needs its own scorer fits and columns, built as A7p's are (a split's scorer never trains on its held-out dataset).
- **zspks** (zspk plus the A7 head before its propagation) against zspk's card fit of each split on P1g-b's pools (P1g-b's scr-p1gb-zspk, scr-p1gb-zspk-hp), after the seed-null re-call: 1 GAIN (L-hotpotqa musique), 0 LOSS (none). Beside: L-musique metaqa WITHIN (+0.0010), L-musique webqsp WITHIN (+0.0036), L-hotpotqa metaqa WITHIN (+0.0005), L-hotpotqa webqsp WITHIN (+0.0042). Next: A7's columns join the full run's models when it runs on P1g-b's pools: one file, declared before its numbers, after P1g-c's, A5cp's, A7p's and A3vp's verdicts, with the composition P1g-c's section 5 sets; each of its splits needs its own scorer fits and columns, built as A7p's are (a split's scorer never trains on its held-out dataset).

F1: both arms adopted: A7's columns join both models, and F1's one input set holds.

zrckps, each read beside, deciding nothing (FC@5 and hit@1 from `scr-a7p-zrckps` and its -hp; A7's delta and call on P1c2's pools from `scr-a7-zrckps-pair-recall.json`):

| split | dataset | read | delta R@5 [95% CI] | call | FC@5 | hit@1 | A7 on P1c2's pools |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| L-musique | metaqa | in-domain | +0.0022 [+0.0011, +0.0033] | WITHIN | +0.0018 | +0.0115 | +0.0024 WITHIN |
| L-musique | squad | in-domain | -0.0001 [-0.0019, +0.0016] | WITHIN | -0.0001 | -0.0011 | -0.0006 WITHIN |
| L-musique | musique | zero-shot | +0.0114 [+0.0065, +0.0164] | WITHIN (filed GAIN) | +0.0074 | -0.0037 | +0.0012 WITHIN |
| L-musique | hotpotqa | in-domain | +0.0051 [+0.0030, +0.0074] | WITHIN | +0.0084 | +0.0032 | -0.0014 WITHIN |
| L-musique | 2wiki | in-domain | -0.0004 [-0.0017, +0.0011] | WITHIN | -0.0024 | -0.0072 | +0.0015 WITHIN |
| L-musique | webqsp | zero-shot | +0.0194 [+0.0092, +0.0296] | GAIN | +0.0106 | +0.0120 | +0.0221 GAIN |
| L-hotpotqa | metaqa | in-domain | +0.0044 [+0.0033, +0.0056] | WITHIN | +0.0039 | +0.0149 | -0.0002 WITHIN |
| L-hotpotqa | squad | in-domain | +0.0017 [-0.0003, +0.0035] | WITHIN | +0.0017 | +0.0012 | +0.0003 WITHIN |
| L-hotpotqa | musique | in-domain | +0.0003 [-0.0050, +0.0060] | WITHIN | -0.0025 | -0.0070 | -0.0008 WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | +0.0076 [+0.0045, +0.0107] | WITHIN (filed GAIN) | +0.0157 | -0.0032 | +0.0038 WITHIN |
| L-hotpotqa | 2wiki | in-domain | -0.0007 [-0.0022, +0.0009] | WITHIN | -0.0036 | -0.0061 | -0.0050 WITHIN |
| L-hotpotqa | webqsp | zero-shot | +0.0273 [+0.0147, +0.0402] | GAIN | +0.0180 | +0.0193 | -0.0219 WITHIN |

zspks, each read beside, deciding nothing (FC@5 and hit@1 from `scr-a7p-zspks` and its -hp; A7's delta and call on P1c2's pools from `scr-a7-zspks-pair-recall.json`):

| split | dataset | read | delta R@5 [95% CI] | call | FC@5 | hit@1 | A7 on P1c2's pools |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| L-musique | metaqa | in-domain | +0.0010 [-0.0001, +0.0022] | WITHIN | +0.0009 | -0.0020 | +0.0007 WITHIN |
| L-musique | squad | in-domain | +0.0013 [-0.0004, +0.0029] | WITHIN | +0.0013 | -0.0028 | -0.0006 WITHIN |
| L-musique | musique | zero-shot | +0.0012 [-0.0042, +0.0066] | WITHIN | +0.0041 | -0.0232 | +0.0133 WITHIN |
| L-musique | hotpotqa | in-domain | +0.0017 [-0.0006, +0.0040] | WITHIN | +0.0031 | -0.0030 | -0.0009 WITHIN |
| L-musique | 2wiki | in-domain | +0.0044 [+0.0030, +0.0059] | WITHIN | +0.0134 | -0.0068 | -0.0049 WITHIN |
| L-musique | webqsp | zero-shot | +0.0036 [-0.0060, +0.0132] | WITHIN | +0.0047 | +0.0033 | -0.0020 WITHIN |
| L-hotpotqa | metaqa | in-domain | +0.0005 [-0.0006, +0.0016] | WITHIN | +0.0000 | +0.0046 | +0.0003 WITHIN |
| L-hotpotqa | squad | in-domain | +0.0015 [-0.0003, +0.0033] | WITHIN | +0.0015 | -0.0004 | -0.0024 WITHIN |
| L-hotpotqa | musique | in-domain | +0.0084 [+0.0031, +0.0135] | GAIN | +0.0058 | -0.0025 | +0.0072 WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | +0.0049 [+0.0022, +0.0076] | WITHIN | +0.0095 | -0.0016 | +0.0049 WITHIN |
| L-hotpotqa | 2wiki | in-domain | +0.0030 [+0.0015, +0.0046] | WITHIN | +0.0061 | -0.0077 | -0.0010 WITHIN |
| L-hotpotqa | webqsp | zero-shot | +0.0042 [-0.0056, +0.0141] | WITHIN | +0.0053 | +0.0106 | +0.0336 GAIN |

Beside, deciding nothing: musique in domain (L-hotpotqa musique, Claim 1's read): zrckps WITHIN (+0.0003), zspks GAIN (+0.0084).

Scorer fits, beside (A7's on P1c2's looks: docs/A7_S6_SCORER_INPUT.md, section 7): L-musique fold 0: epoch 12, held-out 2.863476 (metaqa 2.8966, squad 3.4025, hotpotqa 2.0993, 2wiki 3.0555), gamma 0.92702, eps 0.001; L-musique fold 1: epoch 12, held-out 2.930246 (metaqa 3.1921, squad 2.9581, hotpotqa 2.5217, 2wiki 3.0491), gamma 0.88949, eps 0.001; L-hotpotqa fold 0: epoch 12, held-out 3.741303 (metaqa 3.0235, squad 3.3310, musique 5.5811, 2wiki 3.0297), gamma 0.90468, eps 0.0011; L-hotpotqa fold 1: epoch 10, held-out 3.785524 (metaqa 3.2414, squad 2.9416, musique 5.9792, 2wiki 2.9798), gamma 0.86853, eps 0.0011.

Disk, beside: builds 11.98 GB, columns 2.10 GB, scorer fits 0.00 GB; each stored build: 2wiki/fit-id 0.65 GB, 5928 rows, 1605.8 s, peak 20.37 GB; hotpotqa/fit-id 0.56 GB, 5930 rows, 1510.4 s, peak 15.56 GB; metaqa/fit-al4 0.54 GB, 2980 rows, 1045.3 s, peak 2.57 GB; metaqa/fit-id 1.64 GB, 5960 rows, 1685.9 s, peak 4.12 GB; metaqa/fit-mg3 1.31 GB, 5960 rows, 1330.6 s, peak 3.51 GB; metaqa/fit-rf 1.07 GB, 2980 rows, 1013.6 s, peak 3.00 GB; metaqa/fit-sp4 0.51 GB, 2980 rows, 1019.1 s, peak 2.57 GB; musique/s1fit-id 3.21 GB, 4601 rows, 9096.6 s, peak 58.27 GB; squad/fit-id 0.11 GB, 5856 rows, 646.7 s, peak 0.91 GB. Lifetime: outputs/a7p/cache holds the stored builds, read by the scorer fits and the fit carves' column items (an s1eval build is made in outputs/a7p/cache/tmp inside its column item and removed there); outputs/a7p/fits holds the scorer fits, read by the column items; outputs/a7p/cols holds the columns, read by the card fits and reads; the clean-up keeps outputs/a7p while an A7p item is queued or running (za7p.py names it), and after that the builds and columns are build folders it may remove (regen) while the fits are kept.

This verdict closes only this form (A7's scorer input, its scorer and columns built again on P1g-b's pools); docs/SCREENS.md, 3b. Which pools the full run uses is P1g-c's call (its section 5): on P1g-b's pools this verdict decides whether the columns join, per arm; on P1c2's pools A7's own verdict decides, and this one is reported beside.
