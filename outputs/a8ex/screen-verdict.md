# A8ex's screen verdict: zrckprqe **ADOPT_FOR_FULL_RUN**, zspxqe **NOT_ADOPTED**

Rule: per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A8ex's re-call of the arm, or not twelve reads (docs/A8EX_QDEPTH_PRIOR_ADOPTED.md, section 4).

- **zrckprqe** (zrckpr plus round 11's depth prior (A8e's), added to its final score) against zrckpr's card fit of each split (scr-r1-zrckpr, scr-r1-zrckpr-hp), after the seed-null re-call: 1 GAIN (L-hotpotqa webqsp), 0 LOSS (none). Beside: L-musique metaqa WITHIN (-0.0014), L-musique webqsp WITHIN (+0.0066), L-hotpotqa metaqa WITHIN (-0.0002), L-hotpotqa webqsp GAIN (+0.0315); A8e on zrckpqe: found. Next: the arm fitted again on P1g-b's pools, then its full run, its own declared file.
- **zspxqe** (zspx plus round 11's depth prior (A8e's), added before its three increments read the score) against zspx's card fit of each split (scr-g5-zspx, scr-g5-zspx-hp), after the seed-null re-call: 0 GAIN (none), 0 LOSS (none). Beside: L-musique metaqa WITHIN (+0.0003), L-musique webqsp WITHIN (+0.0018), L-hotpotqa metaqa WITHIN (+0.0015), L-hotpotqa webqsp WITHIN (-0.0089); A8e on zspkqe: found. Next: A8exk on this model as well (section 5): the other arm's GAIN keeps the depth prior open on both models.

Beside, deciding nothing: musique in-domain (L-hotpotqa musique): zrckprqe WITHIN (-0.0067), zspxqe WITHIN (-0.0069).

Each read beside A8e's on the lean pair, deciding nothing (call and delta R@5 after each re-call):

| read | zrckprqe | zrckpqe (A8e) | zspxqe | zspkqe (A8e) |
| --- | --- | --- | --- | --- |
| L-musique metaqa | WITHIN (-0.0014) | WITHIN (-0.0000) | WITHIN (+0.0003) | WITHIN (+0.0008) |
| L-musique squad | WITHIN (-0.0017) | WITHIN (-0.0008) | WITHIN (+0.0017) | WITHIN (+0.0003) |
| L-musique musique | WITHIN (-0.0002) | WITHIN (+0.0031) | WITHIN (+0.0027) | WITHIN (+0.0084) |
| L-musique hotpotqa | WITHIN (-0.0003) | WITHIN (-0.0068) | WITHIN (-0.0009) | WITHIN (-0.0010) |
| L-musique 2wiki | WITHIN (-0.0029) | WITHIN (-0.0058) | WITHIN (-0.0057) | WITHIN (-0.0042) |
| L-musique webqsp | WITHIN (+0.0066) | WITHIN (+0.0024) | WITHIN (+0.0018) | WITHIN (-0.0007) |
| L-hotpotqa metaqa | WITHIN (-0.0002) | WITHIN (+0.0002) | WITHIN (+0.0015) | WITHIN (-0.0008) |
| L-hotpotqa squad | WITHIN (-0.0005) | WITHIN (-0.0005) | WITHIN (+0.0000) | WITHIN (-0.0024) |
| L-hotpotqa musique | WITHIN (-0.0067) | WITHIN (+0.0002) | WITHIN (-0.0069) | WITHIN (-0.0072) |
| L-hotpotqa hotpotqa | WITHIN (-0.0008) | WITHIN (+0.0003) | WITHIN (-0.0074) | WITHIN (+0.0053) |
| L-hotpotqa 2wiki | WITHIN (-0.0026) | WITHIN (-0.0040) | WITHIN (+0.0029) | WITHIN (+0.0037) |
| L-hotpotqa webqsp | GAIN (+0.0315) | WITHIN (+0.0030) | WITHIN (-0.0089) | WITHIN (+0.0133) |

This verdict closes only this form (round 11's depth prior learned inside the adopted arms, zrckpr and zspx); the prior closes as a model term only when neither arm gains (section 5); docs/SCREENS.md, 3b.
