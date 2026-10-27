# G6b's screen verdict: zg6b **NOT_ADOPTED**

Rule: ADOPT_FOR_FULL_RUN when zg6b's re-call against KG's zspk under KG's zspk seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when KG's base verdict is not BASE_READY, a fit or compare is missing or not G6b's, the fits are not on KG's device, or the re-call does not hold the twelve reads (G6's rule, docs/G6_FREE_GNN.md section 4, on the T4).

- **zg6b** (zg6's network over the edges and without them, mixed per question by a gate on the question's edge-family shares) against KG's zspk on the T4, re-called under KG's zspk seed null: 3 GAIN (L-musique metaqa, L-hotpotqa metaqa, L-hotpotqa musique), 3 LOSS (L-musique webqsp, L-hotpotqa hotpotqa, L-hotpotqa webqsp).
- **Next form** (fixed in section 6 before any number): G6b2, a per-row back-off: the gate also reads the row's own log1p in-degree per family over the edges that reach the layers, so a row the edges barely reach can lean on the twin; one declared file.
- Device: {"cuda": "12.8", "cudnn": 91900, "gpu": "Tesla T4", "image": "kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3", "torch": "2.11.0+cu128"}.

Beside, deciding nothing.

- zg6b against KG's zg6 under zspk's floors: 1 GAIN (L-musique webqsp), 0 LOSS (none).

R@5 per read on the T4; the MLP share = (zrckp - rrf) / (zg6b - rrf) where zg6b's gain over rrf exceeds 0.002:

| read | rrf | zspk | zg6 | zg6b | zrckp (MLP) | MLP share | floor | call | vs zg6 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| L-musique metaqa (in-domain) | 0.0047 | 0.8013 | 0.8127 | 0.8126 | 0.7995 | 0.9838 | 0.0075 | GAIN | WITHIN |
| L-musique squad (in-domain) | 0.9054 | 0.9102 | 0.9120 | 0.9113 | 0.9116 | 1.0508 | 0.0075 | WITHIN | WITHIN |
| L-musique musique (zero-shot) | 0.4734 | 0.5146 | 0.5138 | 0.5052 | 0.5129 | 1.2421 | 0.0138 | WITHIN | WITHIN |
| L-musique hotpotqa (in-domain) | 0.6846 | 0.9086 | 0.9127 | 0.9128 | 0.9041 | 0.9619 | 0.0075 | WITHIN | WITHIN |
| L-musique 2wiki (in-domain) | 0.6079 | 0.9364 | 0.9448 | 0.9424 | 0.9319 | 0.9686 | 0.0075 | WITHIN | WITHIN |
| L-musique webqsp (zero-shot) | 0.0535 | 0.3151 | 0.2398 | 0.2784 | 0.3570 | 1.3495 | 0.0075 | LOSS | GAIN |
| L-hotpotqa metaqa (in-domain) | 0.0047 | 0.8037 | 0.8115 | 0.8115 | 0.8038 | 0.9905 | 0.0075 | GAIN | WITHIN |
| L-hotpotqa squad (in-domain) | 0.9054 | 0.9139 | 0.9122 | 0.9130 | 0.9117 | 0.8289 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa musique (in-domain) | 0.4734 | 0.5754 | 0.5953 | 0.5934 | 0.5743 | 0.8408 | 0.0105 | GAIN | WITHIN |
| L-hotpotqa hotpotqa (zero-shot) | 0.6846 | 0.8550 | 0.8391 | 0.8388 | 0.8504 | 1.0752 | 0.0075 | LOSS | WITHIN |
| L-hotpotqa 2wiki (in-domain) | 0.6079 | 0.9265 | 0.9463 | 0.9443 | 0.9354 | 0.9735 | 0.0219 | WITHIN | WITHIN |
| L-hotpotqa webqsp (zero-shot) | 0.0535 | 0.2640 | 0.2430 | 0.2397 | 0.3289 | 1.4791 | 0.0242 | LOSS | WITHIN |

The gate of each fit's p@swa: g = sigmoid(a . sh + b); its value on a question of one family alone, without edges, and over each dataset's s1eval questions (mean, 10th / 50th / 90th percentile; the share of questions without edges):

- L-musique: a = [0.002477, -0.390783, -0.61809], b = -0.333865; one family alone [0.4179, 0.3264, 0.2785], no edges 0.4173.
  - metaqa: mean 0.3992 (0.3948 / 0.4009 / 0.4027); without edges 0.0000 of 9785.
  - squad: mean 0.3438 (0.3036 / 0.3364 / 0.3962); without edges 0.0000 of 11873.
  - musique: mean 0.3936 (0.3828 / 0.3950 / 0.4023); without edges 0.0000 of 2417.
  - hotpotqa: mean 0.3556 (0.3391 / 0.3546 / 0.3739); without edges 0.0000 of 7405.
  - 2wiki: mean 0.3569 (0.3439 / 0.3564 / 0.3708); without edges 0.0000 of 12576.
  - webqsp: mean 0.3667 (0.3525 / 0.3640 / 0.3871); without edges 0.0000 of 1503.
- L-hotpotqa: a = [0.040536, -0.521961, -0.975598], b = -0.44842; one family alone [0.3994, 0.2748, 0.194], no edges 0.3897.
  - metaqa: mean 0.3697 (0.3629 / 0.3725 / 0.3753); without edges 0.0000 of 9785.
  - squad: mean 0.2905 (0.2319 / 0.2789 / 0.3662); without edges 0.0000 of 11873.
  - musique: mean 0.3635 (0.3476 / 0.3656 / 0.3762); without edges 0.0000 of 2417.
  - hotpotqa: mean 0.3070 (0.2844 / 0.3058 / 0.3325); without edges 0.0000 of 7405.
  - 2wiki: mean 0.3101 (0.2925 / 0.3094 / 0.3291); without edges 0.0000 of 12576.
  - webqsp: mean 0.3208 (0.2992 / 0.3168 / 0.3518); without edges 0.0000 of 1503.

Seconds and memory: zg6b and, in brackets, KG's zg6 of the same split, both on the T4:

| split | train s | read s | peak GPU GB, train / read | peak RAM GB |
|---|---:|---:|---|---:|
| L-musique | 2239 (1942) | 875 (773) | 4.8919 / 5.2354 (4.6552 / 5.2370) | 4.5800 (4.5700) |
| L-hotpotqa | 4347 (3998) | 786 (687) | 10.2888 / 10.2888 (10.2883 / 10.2883) | 4.5500 (4.5800) |
