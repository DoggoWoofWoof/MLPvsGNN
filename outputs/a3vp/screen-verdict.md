# A3vp's screen verdict: zrckpy **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when the arm's re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A3vp's re-call of the arm, or not twelve reads (docs/A3VP_P1GB_POOLS.md, section 4; A3v's rule).

- **zrckpy** (zrckp trained with A3v's two-sided share augmentation of zret's reference) against zrckp's card fit of each split on P1g-b's pools (P1g-b's scr-p1gb-zrckp, scr-p1gb-zrckp-hp), after the seed-null re-call: 0 GAIN (none), 1 LOSS (L-musique webqsp). Beside: L-musique metaqa WITHIN (+0.0012), L-musique squad WITHIN (-0.0010), L-musique musique WITHIN (+0.0054), L-musique webqsp LOSS (-0.0113), L-hotpotqa metaqa WITHIN (+0.0030), L-hotpotqa squad WITHIN (+0.0003), L-hotpotqa musique WITHIN (-0.0058), L-hotpotqa webqsp WITHIN (-0.0181). Next: the share idea stays open on the MLP (it gained on P1c2's pools; docs/SCREENS.md, 3b): A3z's whole-pool z-scores beside zret's on P1g-b's pools, its own declared file.

F1: zrckpy not adopted: the full run's MLP trains as zrckp does.

zrckpy, each read beside, deciding nothing (FC@5 and hit@1 from `scr-a3vp-zrckpy` and its -hp; A3v's delta and call on P1c2's pools from `scr-a3v-zrckpy-pair-recall.json`):

| split | dataset | read | delta R@5 [95% CI] | call | FC@5 | hit@1 | A3v on P1c2's pools |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| L-musique | metaqa | in-domain | +0.0012 [+0.0001, +0.0022] | WITHIN | +0.0015 | +0.0119 | +0.0003 WITHIN |
| L-musique | squad | in-domain | -0.0010 [-0.0028, +0.0008] | WITHIN | -0.0010 | -0.0008 | +0.0015 WITHIN |
| L-musique | musique | zero-shot | +0.0054 [+0.0010, +0.0098] | WITHIN | +0.0054 | +0.0062 | -0.0111 WITHIN |
| L-musique | hotpotqa | in-domain | +0.0007 [-0.0016, +0.0030] | WITHIN | -0.0007 | +0.0065 | -0.0022 WITHIN |
| L-musique | 2wiki | in-domain | -0.0069 [-0.0084, -0.0054] | WITHIN | -0.0192 | +0.0051 | +0.0014 WITHIN |
| L-musique | webqsp | zero-shot | -0.0113 [-0.0206, -0.0014] | LOSS | -0.0173 | -0.0140 | +0.0112 GAIN |
| L-hotpotqa | metaqa | in-domain | +0.0030 [+0.0021, +0.0039] | WITHIN | +0.0038 | +0.0106 | -0.0019 WITHIN |
| L-hotpotqa | squad | in-domain | +0.0003 [-0.0016, +0.0022] | WITHIN | +0.0003 | +0.0013 | +0.0003 WITHIN |
| L-hotpotqa | musique | in-domain | -0.0058 [-0.0114, -0.0003] | WITHIN | -0.0062 | -0.0128 | +0.0004 WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0037 [-0.0067, -0.0007] | WITHIN | -0.0070 | -0.0127 | +0.0038 WITHIN |
| L-hotpotqa | 2wiki | in-domain | -0.0054 [-0.0071, -0.0036] | WITHIN | -0.0118 | +0.0053 | -0.0013 WITHIN |
| L-hotpotqa | webqsp | zero-shot | -0.0181 [-0.0295, -0.0069] | WITHIN (filed LOSS) | -0.0180 | -0.0106 | +0.0036 WITHIN |

zrckpy's reweighting in training (stamped by each fit): L-musique 71,263 of 189,392 question visits, 66,673 on the rich side and 4,590 on the thin side; L-hotpotqa 65,884 of 178,760 question visits, 64,433 on the rich side and 1,451 on the thin side.

Beside, deciding nothing: musique in domain (L-hotpotqa musique, Claim 1's read): zrckpy WITHIN (-0.0058).

This verdict closes only this form (A3v's two-sided rule on the MLP, on P1g-b's pools); docs/SCREENS.md, 3b. Which pools the full run uses is P1g-c's call (its section 5): on P1g-b's pools this verdict decides whether the rule joins the MLP; on P1c2's pools A3v's own verdict decides, and this one is reported beside.
