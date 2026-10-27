# R2's verdict on Kaggle's T4 (R2t): zrckpb **NOT_ADOPTED**, zg6m **NOT_ADOPTED**

Rule: per arm (zrckpb, zg6m): ADOPT_FOR_FULL_RUN when its re-call against its base under its KG seed null on the T4 has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when KG's base verdict is not BASE_READY, a fit or compare of the arm is missing or not R2t's, its fits are not on KG's device, a compare's base column is not that base's read as KG filed it, KG's null or the move's compares fail zkg.py's checks, or one of the arm's re-calls does not hold the twelve reads (R2's rule, docs/R2_BRIDGE_EM.md section 4, on KG's files).

- **zrckpb** (zrckp + R2's bridge head) against KG's zrckp on the T4, re-called under KG's zrckp null on the T4: zrckp's base-build fit trained again with seeds 1 and 2, each against its seed-0 fit: 1 GAIN (L-musique musique), 1 LOSS (L-hotpotqa musique). Next form (fixed in section 6 before any number): R2g, a learned per-question back-off on R2's head: the head times 2 sigmoid(a . c_q + b), c_q the question's own embedding projection and its first-stage top-5 RRF margins, the gate starting at 1; one declared file.
- **zg6m** (zg6 + R2's bridge head on its final score) against KG's zg6 on the T4, re-called under KG's zspk null on the T4, each read's seed-0 R@5 moved to zg6's through KG's compares of zg6 with zspk (D1, D2 and the floors stay zspk's): 0 GAIN (none), 2 LOSS (L-musique webqsp, L-hotpotqa webqsp). Next form (fixed in section 6 before any number): R2e, the edge form: bridge scores only for bridges linked to the row by zlink's edges (B1d: the edge is the bridge), as fixed per-row link flags for the MLP and inside zg6's messages for the GNN; one declared file.
- Device: {"cuda": "12.8", "cudnn": 91900, "gpu": "Tesla T4", "image": "kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3", "torch": "2.11.0+cu128"}.

Beside, deciding nothing.

- zg6m against KG's zspk under zspk's null: 3 GAIN (L-musique metaqa, L-hotpotqa metaqa, L-hotpotqa musique), 3 LOSS (L-musique webqsp, L-hotpotqa hotpotqa, L-hotpotqa webqsp).
- zrckpb, webqsp and metaqa: L-musique metaqa WITHIN (+0.0054), L-musique webqsp WITHIN (+0.0120), L-hotpotqa metaqa WITHIN (+0.0018), L-hotpotqa webqsp WITHIN (+0.0098).
  - zrckpb L-musique: L_br per epoch [4.5957, 4.2847, 4.1908, 4.1139, 4.0383, 3.9672, 3.8958, 3.8313]; the term on 188208 of 189392 training questions (0.9937).
  - zrckpb L-hotpotqa: L_br per epoch [5.2139, 4.8308, 4.6754, 4.5439, 4.424, 4.3184, 4.2246, 4.1438]; the term on 177768 of 178760 training questions (0.9945).
- zg6m, webqsp and metaqa: L-musique metaqa WITHIN (-0.0004), L-musique webqsp LOSS (-0.0216), L-hotpotqa metaqa WITHIN (-0.0002), L-hotpotqa webqsp LOSS (-0.0605).
  - zg6m L-musique: L_br per epoch [4.5935, 4.2826, 4.1889, 4.1115, 4.036, 3.9633, 3.8912, 3.8269]; the term on 188208 of 189392 training questions (0.9937).
  - zg6m L-hotpotqa: L_br per epoch [5.2132, 4.8313, 4.6761, 4.5445, 4.4243, 4.3167, 4.2212, 4.1381]; the term on 177768 of 178760 training questions (0.9945).
- musique in-domain (L-hotpotqa musique): zrckpb LOSS (-0.0133), zg6m WITHIN (-0.0041).
- Zrckpb is called "zrckp + R2's bridge head" until the user rules whether reading the five bridges' vectors keeps it inside the MLP track (R2, section 2).
- R2's N2 null and the 7 October null are not read: their fits are the host's, another device (section 4).

R@5 per read on the T4 (R2t's compares; KG's zrckp, zg6 and zspk as filed); the MLP shares (MLP - rrf) / (GNN - rrf) where the GNN's gain over rrf exceeds 0.002:

| read | rrf | zspk | zg6 | zg6m | zrckp (MLP) | zrckpb | zrckp / zg6 | zrckpb / zg6m | zrckpb call | zg6m call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| L-musique metaqa (in-domain) | 0.0047 | 0.8013 | 0.8127 | 0.8123 | 0.7995 | 0.8050 | 0.9837 | 0.9910 | WITHIN | WITHIN |
| L-musique squad (in-domain) | 0.9054 | 0.9102 | 0.9120 | 0.9106 | 0.9116 | 0.9106 | 0.9394 | 1.0000 | WITHIN | WITHIN |
| L-musique musique (zero-shot) | 0.4734 | 0.5146 | 0.5138 | 0.5244 | 0.5129 | 0.5221 | 0.9777 | 0.9549 | GAIN | WITHIN |
| L-musique hotpotqa (in-domain) | 0.6846 | 0.9086 | 0.9127 | 0.9143 | 0.9041 | 0.9063 | 0.9623 | 0.9652 | WITHIN | WITHIN |
| L-musique 2wiki (in-domain) | 0.6079 | 0.9364 | 0.9448 | 0.9417 | 0.9319 | 0.9346 | 0.9617 | 0.9787 | WITHIN | WITHIN |
| L-musique webqsp (zero-shot) | 0.0535 | 0.3151 | 0.2398 | 0.2182 | 0.3570 | 0.3689 | 1.6291 | 1.9150 | WITHIN | LOSS |
| L-hotpotqa metaqa (in-domain) | 0.0047 | 0.8037 | 0.8115 | 0.8113 | 0.8038 | 0.8056 | 0.9905 | 0.9929 | WITHIN | WITHIN |
| L-hotpotqa squad (in-domain) | 0.9054 | 0.9139 | 0.9122 | 0.9079 | 0.9117 | 0.9084 | 0.9265 | 1.2000 | WITHIN | WITHIN |
| L-hotpotqa musique (in-domain) | 0.4734 | 0.5754 | 0.5953 | 0.5912 | 0.5743 | 0.5610 | 0.8277 | 0.7436 | LOSS | WITHIN |
| L-hotpotqa hotpotqa (zero-shot) | 0.6846 | 0.8550 | 0.8391 | 0.8380 | 0.8504 | 0.8454 | 1.0731 | 1.0482 | WITHIN | WITHIN |
| L-hotpotqa 2wiki (in-domain) | 0.6079 | 0.9265 | 0.9463 | 0.9434 | 0.9354 | 0.9340 | 0.9678 | 0.9720 | WITHIN | WITHIN |
| L-hotpotqa webqsp (zero-shot) | 0.0535 | 0.2640 | 0.2430 | 0.1825 | 0.3289 | 0.3387 | 1.4533 | 2.2109 | WITHIN | LOSS |

Each arm minus its KG base: R@5 with its 95% question-bootstrap interval, FC@5 and hit@1 (R2t's compares; the call at 0.0075, before the re-call):

| arm, split | dataset | read | delta R@5 [95% CI] | delta FC@5 | delta hit@1 | call (0.0075) |
|---|---|---|---|---|---|---|
| zrckpb L-musique | metaqa | in-domain | +0.0054 [+0.0042, +0.0067] | +0.0052 | +0.0232 | WITHIN |
| zrckpb L-musique | squad | in-domain | -0.0010 [-0.0031, +0.0010] | -0.0010 | -0.0007 | WITHIN |
| zrckpb L-musique | musique | zero-shot | +0.0091 [+0.0035, +0.0149] | +0.0103 | -0.0261 | GAIN |
| zrckpb L-musique | hotpotqa | in-domain | +0.0022 [-0.0003, +0.0048] | +0.0054 | -0.0161 | WITHIN |
| zrckpb L-musique | 2wiki | in-domain | +0.0026 [+0.0011, +0.0041] | +0.0078 | -0.0093 | WITHIN |
| zrckpb L-musique | webqsp | zero-shot | +0.0120 [+0.0014, +0.0227] | +0.0080 | +0.0333 | GAIN |
| zrckpb L-hotpotqa | metaqa | in-domain | +0.0018 [+0.0006, +0.0031] | +0.0009 | +0.0050 | WITHIN |
| zrckpb L-hotpotqa | squad | in-domain | -0.0033 [-0.0056, -0.0011] | -0.0033 | -0.0067 | WITHIN |
| zrckpb L-hotpotqa | musique | in-domain | -0.0133 [-0.0199, -0.0068] | -0.0116 | -0.0281 | LOSS |
| zrckpb L-hotpotqa | hotpotqa | zero-shot | -0.0050 [-0.0083, -0.0016] | -0.0078 | -0.0117 | WITHIN |
| zrckpb L-hotpotqa | 2wiki | in-domain | -0.0014 [-0.0030, +0.0002] | -0.0007 | -0.0093 | WITHIN |
| zrckpb L-hotpotqa | webqsp | zero-shot | +0.0098 [-0.0020, +0.0212] | +0.0067 | +0.0086 | WITHIN |
| zg6m L-musique | metaqa | in-domain | -0.0004 [-0.0015, +0.0007] | -0.0020 | -0.0028 | WITHIN |
| zg6m L-musique | squad | in-domain | -0.0014 [-0.0035, +0.0006] | -0.0014 | -0.0007 | WITHIN |
| zg6m L-musique | musique | zero-shot | +0.0107 [+0.0053, +0.0163] | +0.0087 | -0.0294 | GAIN |
| zg6m L-musique | hotpotqa | in-domain | +0.0016 [-0.0003, +0.0037] | +0.0034 | -0.0076 | WITHIN |
| zg6m L-musique | 2wiki | in-domain | -0.0031 [-0.0045, -0.0017] | -0.0097 | -0.0083 | WITHIN |
| zg6m L-musique | webqsp | zero-shot | -0.0216 [-0.0393, -0.0047] | -0.0173 | -0.0166 | LOSS |
| zg6m L-hotpotqa | metaqa | in-domain | -0.0002 [-0.0013, +0.0009] | -0.0008 | -0.0026 | WITHIN |
| zg6m L-hotpotqa | squad | in-domain | -0.0043 [-0.0066, -0.0020] | -0.0043 | -0.0067 | WITHIN |
| zg6m L-hotpotqa | musique | in-domain | -0.0041 [-0.0100, +0.0024] | -0.0062 | -0.0236 | WITHIN |
| zg6m L-hotpotqa | hotpotqa | zero-shot | -0.0011 [-0.0043, +0.0022] | +0.0035 | +0.0030 | WITHIN |
| zg6m L-hotpotqa | 2wiki | in-domain | -0.0030 [-0.0045, -0.0014] | -0.0103 | +0.0043 | WITHIN |
| zg6m L-hotpotqa | webqsp | zero-shot | -0.0605 [-0.0778, -0.0424] | -0.0479 | -0.0459 | LOSS |

zg6m minus KG's zspk, beside (R2t's compares; the call at 0.0075, before the re-call):

| arm, split | dataset | read | delta R@5 [95% CI] | delta FC@5 | delta hit@1 | call (0.0075) |
|---|---|---|---|---|---|---|
| zg6m L-musique | metaqa | in-domain | +0.0110 [+0.0092, +0.0126] | +0.0112 | +0.0369 | GAIN |
| zg6m L-musique | squad | in-domain | +0.0003 [-0.0021, +0.0028] | +0.0003 | +0.0022 | WITHIN |
| zg6m L-musique | musique | zero-shot | +0.0098 [+0.0028, +0.0168] | +0.0165 | -0.0174 | GAIN |
| zg6m L-musique | hotpotqa | in-domain | +0.0057 [+0.0035, +0.0081] | +0.0103 | +0.0066 | WITHIN |
| zg6m L-musique | 2wiki | in-domain | +0.0053 [+0.0037, +0.0069] | +0.0148 | +0.0043 | WITHIN |
| zg6m L-musique | webqsp | zero-shot | -0.0969 [-0.1184, -0.0747] | -0.0752 | -0.0585 | LOSS |
| zg6m L-hotpotqa | metaqa | in-domain | +0.0076 [+0.0059, +0.0093] | +0.0084 | +0.0249 | GAIN |
| zg6m L-hotpotqa | squad | in-domain | -0.0060 [-0.0084, -0.0036] | -0.0060 | -0.0013 | WITHIN |
| zg6m L-hotpotqa | musique | in-domain | +0.0158 [+0.0089, +0.0224] | +0.0265 | -0.0079 | GAIN |
| zg6m L-hotpotqa | hotpotqa | zero-shot | -0.0169 [-0.0209, -0.0128] | -0.0146 | -0.0300 | LOSS |
| zg6m L-hotpotqa | 2wiki | in-domain | +0.0168 [+0.0147, +0.0189] | +0.0513 | -0.0067 | GAIN |
| zg6m L-hotpotqa | webqsp | zero-shot | -0.0815 [-0.1026, -0.0611] | -0.0625 | -0.0373 | LOSS |

Seconds and memory: each arm, and in brackets KG's base of the same split, all on the T4:

| arm | split | train s | read s | peak GPU GB, train / read | peak RAM GB |
|---|---|---:|---:|---|---:|
| zrckpb | L-musique | 1107 (960) | 298 (291) | 3.6978 / 4.4335 (3.6978 / 4.4323) | 4.5400 (4.6000) |
| zrckpb | L-hotpotqa | 1165 (963) | 315 (279) | 5.5511 / 5.5511 (5.5511 / 5.5511) | 4.6400 (4.5900) |
| zg6m | L-musique | 2124 (1942) | 849 (773) | 4.6564 / 5.2391 (4.6552 / 5.2370) | 4.6200 (4.5700) |
| zg6m | L-hotpotqa | 4166 (3998) | 714 (687) | 10.2895 / 10.2895 (10.2883 / 10.2883) | 4.5800 (4.5800) |
