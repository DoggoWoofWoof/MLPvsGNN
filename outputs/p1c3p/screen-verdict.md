# P1c3p's screen verdict: **ADOPT_FOR_FULL_RUN**

Rule: ADOPT_FOR_FULL_RUN when zrckp's re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when the re-call is missing, not P1c3p's, or not twelve reads (docs/P1C3P_POINTWISE_HEAD.md, section 4).

zrckp (161 head parameters, the row's own eight KT columns) against zrc on today's pools, after the seed-null re-call: 7 GAIN (L-musique metaqa, L-musique 2wiki, L-musique webqsp, L-hotpotqa metaqa, L-hotpotqa musique, L-hotpotqa 2wiki, L-hotpotqa webqsp), 0 LOSS (none).

Beside, deciding nothing: zrckp minus zrck (P1c3's head with the z-scored score), R@5 / FC@5 / hit@1, with the 95% question-bootstrap interval of the R@5 difference.

| split | dataset | read | delta R@5 [95% CI] | delta FC@5 | delta hit@1 | call (0.0075 floor) |
|---|---|---|---|---|---|---|
| L-musique | metaqa | in-domain | +0.0012 [+0.0002, +0.0021] | -0.0006 | +0.0040 | WITHIN |
| L-musique | squad | in-domain | -0.0001 [-0.0019, +0.0017] | -0.0001 | +0.0027 | WITHIN |
| L-musique | musique | zero-shot | +0.0062 [-0.0001, +0.0123] | +0.0070 | -0.0149 | WITHIN |
| L-musique | hotpotqa | in-domain | +0.0003 [-0.0017, +0.0022] | -0.0008 | +0.0005 | WITHIN |
| L-musique | 2wiki | in-domain | -0.0018 [-0.0033, -0.0004] | -0.0041 | +0.0024 | WITHIN |
| L-musique | webqsp | zero-shot | -0.0129 [-0.0230, -0.0028] | -0.0020 | -0.0047 | LOSS |
| L-hotpotqa | metaqa | in-domain | -0.0008 [-0.0016, +0.0000] | -0.0013 | -0.0050 | WITHIN |
| L-hotpotqa | squad | in-domain | +0.0006 [-0.0014, +0.0024] | +0.0006 | +0.0016 | WITHIN |
| L-hotpotqa | musique | in-domain | +0.0050 [-0.0012, +0.0110] | +0.0017 | -0.0219 | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0051 [-0.0087, -0.0016] | -0.0113 | +0.0080 | WITHIN |
| L-hotpotqa | 2wiki | in-domain | +0.0222 [+0.0200, +0.0243] | +0.0647 | -0.0031 | GAIN |
| L-hotpotqa | webqsp | zero-shot | -0.0211 [-0.0319, -0.0105] | -0.0180 | -0.0053 | LOSS |
