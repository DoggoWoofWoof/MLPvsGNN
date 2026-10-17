# G5's screen verdict: zspx **ADOPT_FOR_FULL_RUN**

Rule: ADOPT_FOR_FULL_RUN when zspx's re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not G5's, or not twelve reads (docs/G5_COMBINED_GNN.md, section 3).

- **zspx** (zspk with its step end to end, G3's residual message and G4's second step) against zspk's card fit of each split under lean_gpu's loss (scr-p1c3-zspk, scr-p1c3-zspk-hp), after the seed-null re-call: 2 GAIN (L-hotpotqa musique, L-hotpotqa webqsp), 0 LOSS (none). Beside: L-musique metaqa WITHIN (+0.0021), L-musique webqsp WITHIN (-0.0025), L-hotpotqa metaqa WITHIN (-0.0000), L-hotpotqa webqsp GAIN (+0.0235).

Beside, deciding nothing: R@5 difference against zspk per read, zspx and each single change.

| read | zspx | zspr | zspk2 | zspkg |
| --- | --- | --- | --- | --- |
| L-musique metaqa | +0.0021 WITHIN | +0.0012 WITHIN | +0.0012 WITHIN | +0.0006 WITHIN |
| L-musique squad | -0.0022 WITHIN | -0.0004 WITHIN | +0.0018 WITHIN | -0.0003 WITHIN |
| L-musique musique | +0.0061 WITHIN | +0.0114 WITHIN | -0.0022 WITHIN | -0.0012 WITHIN |
| L-musique hotpotqa | -0.0007 WITHIN | +0.0003 WITHIN | +0.0004 WITHIN | +0.0003 WITHIN |
| L-musique 2wiki | -0.0032 WITHIN | -0.0018 WITHIN | -0.0003 WITHIN | -0.0028 WITHIN |
| L-musique webqsp | -0.0025 WITHIN | -0.0038 WITHIN | -0.0081 WITHIN | -0.0067 WITHIN |
| L-hotpotqa metaqa | -0.0000 WITHIN | +0.0003 WITHIN | +0.0000 WITHIN | +0.0004 WITHIN |
| L-hotpotqa squad | -0.0006 WITHIN | -0.0003 WITHIN | -0.0003 WITHIN | -0.0003 WITHIN |
| L-hotpotqa musique | +0.0147 GAIN | +0.0096 GAIN | +0.0115 GAIN | +0.0093 GAIN |
| L-hotpotqa hotpotqa | +0.0048 WITHIN | -0.0029 WITHIN | +0.0036 WITHIN | +0.0024 WITHIN |
| L-hotpotqa 2wiki | -0.0020 WITHIN | +0.0025 WITHIN | +0.0054 WITHIN | -0.0010 WITHIN |
| L-hotpotqa webqsp | +0.0235 GAIN | +0.0132 WITHIN | +0.0030 WITHIN | +0.0152 WITHIN |
