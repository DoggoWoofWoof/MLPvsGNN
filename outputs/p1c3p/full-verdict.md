# P1c3p's full run (docs/P1C3P_FULL_RUN.md): **ADOPT**

Rule: ADOPT when zrckp's re-grade under the seed null has no LOSS on any of its 36 reads and at least one primary GAIN; NOT_ADOPTED otherwise; INCOMPLETE when the re-grade is missing or not the declared one (docs/P1C3P_FULL_RUN.md, section 4).

zrckp (161 head parameters, the row's own eight KT columns) against zrc on today's pools, after the re-grade under the seed null: 5 primary GAIN (J5 metaqa, J5 2wiki, J5 webqsp, L-metaqa metaqa, L-2wiki 2wiki), 20 GAIN in all, 0 LOSS (none); the re-grade's own verdict ADOPT, deciding nothing.

Beside, deciding nothing: zrckp minus zrck (P1c3's head with the z-scored score), R@5 / FC@5 / hit@1, with the 95% question-bootstrap interval of the R@5 difference.

| split | dataset | read | delta R@5 [95% CI] | delta FC@5 | delta hit@1 | call (0.0075 floor) |
|---|---|---|---|---|---|---|
| J5 | metaqa | in-domain | -0.0003 [-0.0012, +0.0006] | +0.0002 | -0.0001 | WITHIN |
| J5 | squad | in-domain | +0.0004 [-0.0017, +0.0024] | +0.0004 | +0.0030 | WITHIN |
| J5 | musique | in-domain | +0.0040 [-0.0016, +0.0096] | +0.0029 | -0.0190 | WITHIN |
| J5 | hotpotqa | in-domain | +0.0028 [+0.0004, +0.0051] | +0.0045 | -0.0109 | WITHIN |
| J5 | 2wiki | in-domain | +0.0035 [+0.0017, +0.0053] | +0.0150 | -0.0054 | WITHIN |
| J5 | webqsp | zero-shot | +0.0011 [-0.0084, +0.0110] | -0.0027 | +0.0007 | WITHIN |
| L-metaqa | metaqa | zero-shot | +0.0459 [+0.0407, +0.0507] | +0.0407 | -0.0048 | GAIN |
| L-metaqa | squad | in-domain | +0.0002 [-0.0016, +0.0020] | +0.0002 | +0.0025 | WITHIN |
| L-metaqa | musique | in-domain | +0.0054 [+0.0006, +0.0106] | +0.0070 | +0.0008 | WITHIN |
| L-metaqa | hotpotqa | in-domain | +0.0010 [-0.0010, +0.0032] | +0.0026 | -0.0038 | WITHIN |
| L-metaqa | 2wiki | in-domain | -0.0030 [-0.0045, -0.0015] | -0.0056 | -0.0090 | WITHIN |
| L-metaqa | webqsp | zero-shot | -0.0069 [-0.0131, -0.0006] | -0.0067 | -0.0060 | WITHIN |
| L-squad | metaqa | in-domain | +0.0003 [-0.0009, +0.0014] | -0.0003 | +0.0034 | WITHIN |
| L-squad | squad | zero-shot | +0.0026 [+0.0002, +0.0048] | +0.0026 | +0.0006 | WITHIN |
| L-squad | musique | in-domain | +0.0006 [-0.0054, +0.0064] | -0.0017 | -0.0128 | WITHIN |
| L-squad | hotpotqa | in-domain | +0.0010 [-0.0011, +0.0031] | +0.0024 | -0.0157 | WITHIN |
| L-squad | 2wiki | in-domain | +0.0019 [+0.0004, +0.0036] | +0.0085 | -0.0060 | WITHIN |
| L-squad | webqsp | zero-shot | -0.0066 [-0.0175, +0.0038] | -0.0033 | +0.0033 | WITHIN |
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
| L-2wiki | metaqa | in-domain | +0.0018 [+0.0008, +0.0027] | +0.0020 | +0.0046 | WITHIN |
| L-2wiki | squad | in-domain | +0.0012 [-0.0006, +0.0029] | +0.0012 | +0.0032 | WITHIN |
| L-2wiki | musique | in-domain | +0.0020 [-0.0028, +0.0067] | +0.0037 | -0.0050 | WITHIN |
| L-2wiki | hotpotqa | in-domain | +0.0014 [-0.0009, +0.0038] | +0.0036 | -0.0009 | WITHIN |
| L-2wiki | 2wiki | zero-shot | -0.0034 [-0.0056, -0.0014] | -0.0055 | -0.0047 | WITHIN |
| L-2wiki | webqsp | zero-shot | +0.0096 [+0.0009, +0.0183] | +0.0100 | +0.0106 | GAIN |
