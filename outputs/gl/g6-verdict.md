# G6's verdict on the L40S (GL): zg6 **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when zg6's re-call under U2d's zspk seed null (N2's design on the L40S) has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when that re-call is missing, is not GL's re-call of zg6 under U2d's zspk null, or does not hold the twelve reads (docs/G6_FREE_GNN.md, section 4, with the files of docs/GL_GPU_LINE_ON_L40S.md, section 2).

- **zg6** (the free GNN: its own encoder, 3 layers of 64-wide hidden-state message passing, an unbounded readout, no MLP score) against zspk's seed-0 fit on U2d's base build (scr-u2d-base-zspk, scr-u2d-base-zspk-hp) on Modal's L40S: the same pools, KT input, card settings and device, after the re-call under U2d's zspk null on the L40S: zspk's base-build fit trained again with seeds 1 and 2, each against its seed-0 fit (N2's design, on the device the arms ran on): 2 GAIN (L-musique metaqa, L-hotpotqa musique), 3 LOSS (L-musique webqsp, L-hotpotqa hotpotqa, L-hotpotqa webqsp).
- **Next form** (fixed in docs/G6_FREE_GNN.md, section 5, before any number): G6b, a per-question back-off: zg6's score mixed with its twin's through a learned gate on the question's edge-family shares, at equal weight at the start; one declared file.
- Device: {"gpu": "NVIDIA L40S", "image": "u2d-gpu:py3.13:torch2.8.0+cu128:numpy2.3.2", "torch": "2.8.0+cu128", "cuda": "12.8", "cudnn": 91002}.

Beside, deciding nothing.

- The twin: zg6t has no L40S fit yet: it runs on workspace 3 from 1 November (section 4), filed beside.
- The 7 October re-call is not computed: its re-base reads the host's fits, another device (section 2).

R@5 per read, all on the L40S except the last column; the MLP share = (zrckp - rrf) / (zg6 - rrf) where zg6's gain over rrf exceeds 0.002:

| read | rrf | zspk | zg6 | zrckp (MLP) | MLP share | floor | call (filed -> re-call) | zspk, host card |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| L-musique metaqa (in-domain) | 0.0047 | 0.8021 | 0.8123 | 0.8008 | 0.9858 | 0.0075 | GAIN | 0.8009 |
| L-musique squad (in-domain) | 0.9054 | 0.9118 | 0.9106 | 0.9121 | 1.2885 | 0.0075 | WITHIN | 0.9114 |
| L-musique musique (zero-shot) | 0.4734 | 0.5143 | 0.5207 | 0.5153 | 0.8858 | 0.0089 | WITHIN | 0.5012 |
| L-musique hotpotqa (in-domain) | 0.6846 | 0.9088 | 0.9118 | 0.9055 | 0.9723 | 0.0075 | WITHIN | 0.9080 |
| L-musique 2wiki (in-domain) | 0.6079 | 0.9346 | 0.9416 | 0.9345 | 0.9787 | 0.0075 | WITHIN | 0.9383 |
| L-musique webqsp (zero-shot) | 0.0535 | 0.3162 | 0.2368 | 0.3561 | 1.6508 | 0.0105 | LOSS | 0.3229 |
| L-hotpotqa metaqa (in-domain) | 0.0047 | 0.8043 | 0.8112 | 0.8033 | 0.9902 | 0.0075 | WITHIN | 0.8040 |
| L-hotpotqa squad (in-domain) | 0.9054 | 0.9117 | 0.9118 | 0.9124 | 1.0937 | 0.0075 | WITHIN | 0.9135 |
| L-hotpotqa musique (in-domain) | 0.4734 | 0.5800 | 0.5937 | 0.5697 | 0.8005 | 0.0107 | GAIN | 0.5754 |
| L-hotpotqa hotpotqa (zero-shot) | 0.6846 | 0.8512 | 0.8309 | 0.8538 | 1.1565 | 0.0145 | LOSS | 0.8499 |
| L-hotpotqa 2wiki (in-domain) | 0.6079 | 0.9290 | 0.9391 | 0.9303 | 0.9734 | 0.0139 | GAIN -> WITHIN | 0.9333 |
| L-hotpotqa webqsp (zero-shot) | 0.0535 | 0.2773 | 0.2189 | 0.3370 | 1.7140 | 0.0405 | LOSS | 0.2680 |

zg6 minus zspk: R@5 with its 95% question-bootstrap interval, FC@5 and hit@1 (U2d's compares; the call at 0.0075, before the re-call):

| split | dataset | read | delta R@5 [95% CI] | delta FC@5 | delta hit@1 | call (0.0075) |
|---|---|---|---|---|---|---|
| L-musique | metaqa | in-domain | +0.0102 [+0.0085, +0.0118] | +0.0118 | +0.0355 | GAIN |
| L-musique | squad | in-domain | -0.0012 [-0.0033, +0.0010] | -0.0012 | +0.0061 | WITHIN |
| L-musique | musique | zero-shot | +0.0063 [-0.0010, +0.0131] | +0.0120 | +0.0116 | WITHIN |
| L-musique | hotpotqa | in-domain | +0.0030 [+0.0005, +0.0053] | +0.0076 | +0.0059 | WITHIN |
| L-musique | 2wiki | in-domain | +0.0070 [+0.0055, +0.0087] | +0.0192 | +0.0024 | WITHIN |
| L-musique | webqsp | zero-shot | -0.0795 [-0.1003, -0.0596] | -0.0599 | -0.0519 | LOSS |
| L-hotpotqa | metaqa | in-domain | +0.0068 [+0.0054, +0.0083] | +0.0083 | +0.0251 | WITHIN |
| L-hotpotqa | squad | in-domain | +0.0001 [-0.0020, +0.0023] | +0.0001 | +0.0034 | WITHIN |
| L-hotpotqa | musique | in-domain | +0.0138 [+0.0070, +0.0204] | +0.0236 | +0.0393 | GAIN |
| L-hotpotqa | hotpotqa | zero-shot | -0.0203 [-0.0248, -0.0161] | -0.0221 | -0.0408 | LOSS |
| L-hotpotqa | 2wiki | in-domain | +0.0101 [+0.0080, +0.0123] | +0.0297 | +0.0091 | GAIN |
| L-hotpotqa | webqsp | zero-shot | -0.0584 [-0.0783, -0.0381] | -0.0406 | -0.0067 | LOSS |
