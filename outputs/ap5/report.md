# AP5: label precision by question type (2026-11-17T14:56:40Z)

Declared in `docs/AP5_TYPED_LABEL_PRECISION.md`. AP4's filed ledgers split by the datasets' own question types. Precision in %;
label pairs are paired differences in points (95%, 2,000 resamples of the type's questions, seed 0).

**Verdict (D83E decides): TYPED** - 2 of 4 datasets FLIP at the top1 anchor (need 2). Beside: {'B1V': {'hotpotqa': 'NO_FLIP', '2wiki': 'FLIP', 'musique': 'NO_FLIP'}}.

## D83E hotpotqa (s1eval): {'bridge': 5918, 'comparison': 1487}

top1: NO_FLIP [] | found: NO_FLIP []

| type | kind | questions | all golds in top 5 | unit | pairs/anchor | precision | lift | vs any |
|---|---|---|---|---|---|---|---|---|
| bridge | found | 5520 | 0.39366 | any | 27.5606 | +1.4 [+1.3, +1.4] | +16.1 [+15.6, +16.5] | -  |
| bridge | found | 5520 | 0.39366 | label:R.mention | 14.812 | +2.3 [+2.2, +2.5] | +27.2 [+26.1, +28.4] | +0.9 [+0.9, +1.0] ABOVE |
| bridge | found | 5520 | 0.39366 | label:R.name_comention | 11.8731 | +1.1 [+1.0, +1.1] | +12.7 [+11.9, +13.4] | -0.3 [-0.4, -0.2] BELOW |
| bridge | found | 5520 | 0.39366 | label:R.title_comention | 0.8669 | +7.0 [+6.4, +7.7] | +82.3 [+74.7, +90.4] | +5.7 [+5.0, +6.4] ABOVE |
| bridge | found | 5520 | 0.39366 | label:K.knn | 4.474 | +1.7 [+1.6, +1.8] | +20.0 [+18.5, +21.6] | +0.3 [+0.2, +0.5] ABOVE |
| bridge | top1 | 5918 | 0.36799 | any | 25.3792 | +1.8 [+1.7, +1.9] | +18.7 [+18.1, +19.3] | -  |
| bridge | top1 | 5918 | 0.36799 | label:R.mention | 12.3897 | +3.3 [+3.1, +3.4] | +33.9 [+32.3, +35.7] | +1.5 [+1.3, +1.6] ABOVE |
| bridge | top1 | 5918 | 0.36799 | label:R.name_comention | 11.8682 | +1.3 [+1.3, +1.4] | +14.0 [+13.2, +14.8] | -0.5 [-0.5, -0.4] BELOW |
| bridge | top1 | 5918 | 0.36799 | label:R.title_comention | 0.7389 | +9.3 [+8.4, +10.3] | +97.3 [+87.9, +107.3] | +7.5 [+6.6, +8.5] ABOVE |
| bridge | top1 | 5918 | 0.36799 | label:K.knn | 4.1984 | +2.3 [+2.1, +2.5] | +23.7 [+21.9, +25.6] | +0.5 [+0.3, +0.6] ABOVE |
| comparison | found | 1479 | 0.8073 | any | 34.488 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -  |
| comparison | found | 1479 | 0.8073 | label:R.mention | 25.9854 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | found | 1479 | 0.8073 | label:R.name_comention | 9.1416 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | found | 1479 | 0.8073 | label:R.title_comention | 0.9513 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | found | 1479 | 0.8073 | label:K.knn | 5.2876 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | top1 | 1487 | 0.80296 | any | 32.0962 | +0.0 [+0.0, +0.0] | +0.8 [+0.4, +1.3] | -  |
| comparison | top1 | 1487 | 0.80296 | label:R.mention | 23.7108 | +0.0 [+0.0, +0.0] | +0.7 [+0.3, +1.3] | -0.0 [-0.0, +0.0] LEVEL |
| comparison | top1 | 1487 | 0.80296 | label:R.name_comention | 8.8063 | +0.0 [+0.0, +0.1] | +1.0 [+0.2, +2.0] | +0.0 [-0.0, +0.0] LEVEL |
| comparison | top1 | 1487 | 0.80296 | label:R.title_comention | 0.885 | +0.2 [+0.0, +0.4] | +4.8 [+0.0, +12.4] | +0.1 [-0.0, +0.4] LEVEL |
| comparison | top1 | 1487 | 0.80296 | label:K.knn | 5.0693 | +0.1 [+0.0, +0.1] | +1.7 [+0.4, +3.5] | +0.0 [-0.0, +0.1] LEVEL |

| type | kind | label pair | difference | call |
|---|---|---|---|---|
| bridge | found | R.mention - R.name_comention | +1.2 [+1.1, +1.4] | ABOVE |
| bridge | found | R.mention - R.title_comention | -4.7 [-5.4, -4.0] | BELOW |
| bridge | found | R.mention - K.knn | +0.6 [+0.5, +0.8] | ABOVE |
| bridge | found | R.name_comention - R.title_comention | -5.9 [-6.7, -5.3] | BELOW |
| bridge | found | R.name_comention - K.knn | -0.6 [-0.8, -0.5] | BELOW |
| bridge | found | R.title_comention - K.knn | +5.3 [+4.7, +6.0] | ABOVE |
| bridge | top1 | R.mention - R.name_comention | +1.9 [+1.7, +2.1] | ABOVE |
| bridge | top1 | R.mention - R.title_comention | -6.1 [-7.0, -5.2] | BELOW |
| bridge | top1 | R.mention - K.knn | +1.0 [+0.8, +1.2] | ABOVE |
| bridge | top1 | R.name_comention - R.title_comention | -8.0 [-9.0, -7.1] | BELOW |
| bridge | top1 | R.name_comention - K.knn | -0.9 [-1.1, -0.8] | BELOW |
| bridge | top1 | R.title_comention - K.knn | +7.1 [+6.2, +8.0] | ABOVE |
| comparison | found | R.mention - R.name_comention | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | found | R.mention - R.title_comention | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | found | R.mention - K.knn | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | found | R.name_comention - R.title_comention | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | found | R.name_comention - K.knn | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | found | R.title_comention - K.knn | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | top1 | R.mention - R.name_comention | -0.0 [-0.0, +0.0] | LEVEL |
| comparison | top1 | R.mention - R.title_comention | -0.1 [-0.4, +0.0] | LEVEL |
| comparison | top1 | R.mention - K.knn | -0.0 [-0.1, +0.0] | LEVEL |
| comparison | top1 | R.name_comention - R.title_comention | -0.1 [-0.4, +0.0] | LEVEL |
| comparison | top1 | R.name_comention - K.knn | -0.0 [-0.1, +0.0] | LEVEL |
| comparison | top1 | R.title_comention - K.knn | +0.1 [-0.1, +0.3] | LEVEL |

## D83E 2wiki (s1eval): {'compositional': 5236, 'comparison': 3040, 'bridge_comparison': 2751, 'inference': 1549}

top1: FLIP [{'pair': 'R.mention - K.knn', 'ABOVE_in': ['bridge_comparison'], 'BELOW_in': ['inference']}] | found: FLIP [{'pair': 'R.mention - K.knn', 'ABOVE_in': ['bridge_comparison'], 'BELOW_in': ['compositional', 'inference']}]

| type | kind | questions | all golds in top 5 | unit | pairs/anchor | precision | lift | vs any |
|---|---|---|---|---|---|---|---|---|
| compositional | found | 5146 | 0.15274 | any | 41.499 | +1.7 [+1.7, +1.8] | +22.5 [+21.9, +23.1] | -  |
| compositional | found | 5146 | 0.15274 | label:R.mention | 20.0919 | +3.2 [+3.1, +3.4] | +42.7 [+41.1, +44.4] | +1.5 [+1.4, +1.6] ABOVE |
| compositional | found | 5146 | 0.15274 | label:R.name_comention | 21.7168 | +1.8 [+1.7, +1.8] | +23.0 [+22.3, +23.8] | +0.0 [-0.0, +0.1] LEVEL |
| compositional | found | 5146 | 0.15274 | label:R.title_comention | 0.751 | +10.0 [+9.0, +11.1] | +132.1 [+119.3, +145.5] | +8.3 [+7.4, +9.4] ABOVE |
| compositional | found | 5146 | 0.15274 | label:K.knn | 3.9413 | +3.9 [+3.6, +4.1] | +51.0 [+47.7, +54.1] | +2.2 [+1.9, +2.4] ABOVE |
| compositional | top1 | 5236 | 0.15011 | any | 36.9163 | +1.9 [+1.9, +2.0] | +25.0 [+24.5, +25.6] | -  |
| compositional | top1 | 5236 | 0.15011 | label:R.mention | 16.4064 | +4.0 [+3.9, +4.1] | +51.3 [+49.9, +52.6] | +2.0 [+2.0, +2.1] ABOVE |
| compositional | top1 | 5236 | 0.15011 | label:R.name_comention | 20.3165 | +1.9 [+1.8, +2.0] | +24.3 [+23.5, +25.2] | -0.1 [-0.1, +0.0] LEVEL |
| compositional | top1 | 5236 | 0.15011 | label:R.title_comention | 0.6515 | +11.8 [+10.6, +12.9] | +151.6 [+135.7, +166.7] | +9.8 [+8.6, +11.0] ABOVE |
| compositional | top1 | 5236 | 0.15011 | label:K.knn | 3.8209 | +4.1 [+3.9, +4.4] | +53.3 [+50.1, +56.7] | +2.2 [+2.0, +2.5] ABOVE |
| comparison | found | 2991 | 0.73119 | any | 30.1587 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -  |
| comparison | found | 2991 | 0.73119 | label:R.mention | 14.6398 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | found | 2991 | 0.73119 | label:R.name_comention | 14.418 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | found | 2991 | 0.73119 | label:R.title_comention | 0.3695 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | found | 2991 | 0.73119 | label:K.knn | 4.226 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | top1 | 3040 | 0.71941 | any | 28.65 | +0.0 [+0.0, +0.0] | +0.4 [+0.2, +0.6] | -  |
| comparison | top1 | 3040 | 0.71941 | label:R.mention | 13.9605 | +0.0 [+0.0, +0.0] | +0.6 [+0.2, +1.1] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | top1 | 3040 | 0.71941 | label:R.name_comention | 13.7289 | +0.0 [+0.0, +0.0] | +0.2 [+0.0, +0.4] | -0.0 [-0.0, +0.0] LEVEL |
| comparison | top1 | 3040 | 0.71941 | label:R.title_comention | 0.3747 | +0.1 [+0.0, +0.3] | +2.9 [+0.0, +9.4] | +0.1 [-0.0, +0.3] LEVEL |
| comparison | top1 | 3040 | 0.71941 | label:K.knn | 4.0977 | +0.0 [+0.0, +0.1] | +1.1 [+0.3, +2.3] | +0.0 [-0.0, +0.1] LEVEL |
| bridge_comparison | found | 2731 | 0.00366 | any | 38.2036 | +2.3 [+2.2, +2.3] | +12.3 [+12.0, +12.6] | -  |
| bridge_comparison | found | 2731 | 0.00366 | label:R.mention | 15.783 | +5.3 [+5.2, +5.5] | +29.1 [+28.3, +29.8] | +3.1 [+3.0, +3.2] ABOVE |
| bridge_comparison | found | 2731 | 0.00366 | label:R.name_comention | 21.6625 | +1.6 [+1.5, +1.7] | +8.6 [+8.3, +9.0] | -0.7 [-0.7, -0.6] BELOW |
| bridge_comparison | found | 2731 | 0.00366 | label:R.title_comention | 0.4275 | +14.3 [+12.7, +15.9] | +77.8 [+69.6, +86.5] | +12.0 [+10.5, +13.6] ABOVE |
| bridge_comparison | found | 2731 | 0.00366 | label:K.knn | 4.3118 | +2.0 [+1.9, +2.2] | +11.1 [+10.1, +12.2] | -0.2 [-0.4, -0.0] BELOW |
| bridge_comparison | top1 | 2751 | 0.00364 | any | 36.4104 | +2.2 [+2.2, +2.3] | +12.0 [+11.7, +12.3] | -  |
| bridge_comparison | top1 | 2751 | 0.00364 | label:R.mention | 15.076 | +5.2 [+5.0, +5.4] | +28.2 [+27.4, +29.1] | +3.0 [+2.9, +3.1] ABOVE |
| bridge_comparison | top1 | 2751 | 0.00364 | label:R.name_comention | 20.6067 | +1.7 [+1.6, +1.7] | +8.9 [+8.4, +9.5] | -0.6 [-0.6, -0.5] BELOW |
| bridge_comparison | top1 | 2751 | 0.00364 | label:R.title_comention | 0.4348 | +13.8 [+11.8, +16.0] | +74.7 [+64.1, +86.4] | +11.6 [+9.6, +13.7] ABOVE |
| bridge_comparison | top1 | 2751 | 0.00364 | label:K.knn | 4.1778 | +2.0 [+1.8, +2.3] | +10.9 [+9.6, +12.4] | -0.2 [-0.4, +0.1] LEVEL |
| inference | found | 1516 | 0.21042 | any | 45.2667 | +1.4 [+1.3, +1.4] | +17.2 [+16.6, +17.8] | -  |
| inference | found | 1516 | 0.21042 | label:R.mention | 21.7459 | +2.5 [+2.3, +2.6] | +30.9 [+29.4, +32.5] | +1.1 [+1.0, +1.2] ABOVE |
| inference | found | 1516 | 0.21042 | label:R.name_comention | 25.0186 | +1.6 [+1.5, +1.7] | +20.1 [+19.1, +21.3] | +0.2 [+0.2, +0.3] ABOVE |
| inference | found | 1516 | 0.21042 | label:R.title_comention | 0.9399 | +7.3 [+6.0, +8.7] | +92.2 [+75.5, +109.2] | +6.0 [+4.7, +7.3] ABOVE |
| inference | found | 1516 | 0.21042 | label:K.knn | 3.6475 | +7.0 [+6.5, +7.6] | +88.6 [+81.7, +95.2] | +5.7 [+5.1, +6.2] ABOVE |
| inference | top1 | 1549 | 0.20594 | any | 42.2034 | +1.5 [+1.5, +1.6] | +18.6 [+17.9, +19.3] | -  |
| inference | top1 | 1549 | 0.20594 | label:R.mention | 19.6617 | +2.7 [+2.6, +2.9] | +33.3 [+31.6, +34.9] | +1.2 [+1.1, +1.3] ABOVE |
| inference | top1 | 1549 | 0.20594 | label:R.name_comention | 23.4861 | +1.8 [+1.6, +1.9] | +21.5 [+20.2, +22.7] | +0.2 [+0.2, +0.3] ABOVE |
| inference | top1 | 1549 | 0.20594 | label:R.title_comention | 0.8799 | +7.6 [+6.2, +9.2] | +93.2 [+76.3, +112.0] | +6.1 [+4.7, +7.6] ABOVE |
| inference | top1 | 1549 | 0.20594 | label:K.knn | 3.734 | +7.5 [+6.8, +8.1] | +91.0 [+83.4, +98.9] | +5.9 [+5.3, +6.5] ABOVE |

| type | kind | label pair | difference | call |
|---|---|---|---|---|
| compositional | found | R.mention - R.name_comention | +1.5 [+1.4, +1.6] | ABOVE |
| compositional | found | R.mention - R.title_comention | -6.8 [-7.8, -5.8] | BELOW |
| compositional | found | R.mention - K.knn | -0.6 [-0.9, -0.4] | BELOW |
| compositional | found | R.name_comention - R.title_comention | -8.3 [-9.3, -7.3] | BELOW |
| compositional | found | R.name_comention - K.knn | -2.1 [-2.4, -1.9] | BELOW |
| compositional | found | R.title_comention - K.knn | +6.2 [+5.2, +7.2] | ABOVE |
| compositional | top1 | R.mention - R.name_comention | +2.1 [+2.0, +2.2] | ABOVE |
| compositional | top1 | R.mention - R.title_comention | -7.8 [-9.0, -6.6] | BELOW |
| compositional | top1 | R.mention - K.knn | -0.2 [-0.4, +0.1] | LEVEL |
| compositional | top1 | R.name_comention - R.title_comention | -9.9 [-11.1, -8.7] | BELOW |
| compositional | top1 | R.name_comention - K.knn | -2.3 [-2.5, -2.0] | BELOW |
| compositional | top1 | R.title_comention - K.knn | +7.6 [+6.4, +8.8] | ABOVE |
| comparison | found | R.mention - R.name_comention | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | found | R.mention - R.title_comention | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | found | R.mention - K.knn | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | found | R.name_comention - R.title_comention | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | found | R.name_comention - K.knn | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | found | R.title_comention - K.knn | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | top1 | R.mention - R.name_comention | +0.0 [+0.0, +0.0] | ABOVE |
| comparison | top1 | R.mention - R.title_comention | -0.1 [-0.3, +0.0] | LEVEL |
| comparison | top1 | R.mention - K.knn | -0.0 [-0.0, +0.0] | LEVEL |
| comparison | top1 | R.name_comention - R.title_comention | -0.1 [-0.3, +0.0] | LEVEL |
| comparison | top1 | R.name_comention - K.knn | -0.0 [-0.1, -0.0] | BELOW |
| comparison | top1 | R.title_comention - K.knn | +0.1 [-0.1, +0.3] | LEVEL |
| bridge_comparison | found | R.mention - R.name_comention | +3.8 [+3.6, +3.9] | ABOVE |
| bridge_comparison | found | R.mention - R.title_comention | -8.9 [-10.5, -7.4] | BELOW |
| bridge_comparison | found | R.mention - K.knn | +3.3 [+3.1, +3.5] | ABOVE |
| bridge_comparison | found | R.name_comention - R.title_comention | -12.7 [-14.3, -11.2] | BELOW |
| bridge_comparison | found | R.name_comention - K.knn | -0.5 [-0.6, -0.3] | BELOW |
| bridge_comparison | found | R.title_comention - K.knn | +12.2 [+10.7, +13.8] | ABOVE |
| bridge_comparison | top1 | R.mention - R.name_comention | +3.6 [+3.4, +3.7] | ABOVE |
| bridge_comparison | top1 | R.mention - R.title_comention | -8.6 [-10.7, -6.6] | BELOW |
| bridge_comparison | top1 | R.mention - K.knn | +3.2 [+2.9, +3.5] | ABOVE |
| bridge_comparison | top1 | R.name_comention - R.title_comention | -12.1 [-14.3, -10.2] | BELOW |
| bridge_comparison | top1 | R.name_comention - K.knn | -0.4 [-0.6, -0.1] | BELOW |
| bridge_comparison | top1 | R.title_comention - K.knn | +11.8 [+9.8, +14.0] | ABOVE |
| inference | found | R.mention - R.name_comention | +0.9 [+0.7, +1.0] | ABOVE |
| inference | found | R.mention - R.title_comention | -4.9 [-6.2, -3.6] | BELOW |
| inference | found | R.mention - K.knn | -4.6 [-5.1, -4.0] | BELOW |
| inference | found | R.name_comention - R.title_comention | -5.7 [-7.1, -4.4] | BELOW |
| inference | found | R.name_comention - K.knn | -5.4 [-6.0, -4.9] | BELOW |
| inference | found | R.title_comention - K.knn | +0.3 [-1.2, +1.8] | LEVEL |
| inference | top1 | R.mention - R.name_comention | +1.0 [+0.8, +1.1] | ABOVE |
| inference | top1 | R.mention - R.title_comention | -4.9 [-6.4, -3.5] | BELOW |
| inference | top1 | R.mention - K.knn | -4.7 [-5.3, -4.1] | BELOW |
| inference | top1 | R.name_comention - R.title_comention | -5.9 [-7.5, -4.5] | BELOW |
| inference | top1 | R.name_comention - K.knn | -5.7 [-6.3, -5.1] | BELOW |
| inference | top1 | R.title_comention - K.knn | +0.2 [-1.4, +1.8] | LEVEL |

## D83E musique (s1eval): {'2': 1252, '3': 760, '4': 405}

top1: NO_FLIP [] | found: NO_FLIP []

| type | kind | questions | all golds in top 5 | unit | pairs/anchor | precision | lift | vs any |
|---|---|---|---|---|---|---|---|---|
| 2 | found | 1106 | 0.27939 | any | 58.6887 | +0.7 [+0.6, +0.8] | +49.1 [+44.1, +54.7] | -  |
| 2 | found | 1106 | 0.27939 | label:R.mention | 40.108 | +0.5 [+0.4, +0.6] | +37.7 [+31.9, +44.3] | -0.2 [-0.2, -0.1] BELOW |
| 2 | found | 1106 | 0.27939 | label:R.name_comention | 18.7157 | +1.7 [+1.6, +1.9] | +124.5 [+114.9, +135.2] | +1.1 [+0.9, +1.2] ABOVE |
| 2 | found | 1106 | 0.27939 | label:R.title_comention | 0.8372 | +3.8 [+2.7, +5.3] | +272.9 [+188.8, +373.6] | +3.1 [+2.0, +4.6] ABOVE |
| 2 | found | 1106 | 0.27939 | label:K.knn | 3.8849 | +2.3 [+1.9, +2.8] | +167.3 [+140.3, +196.1] | +1.7 [+1.3, +2.1] ABOVE |
| 2 | top1 | 1252 | 0.24681 | any | 59.52 | +0.8 [+0.7, +0.9] | +49.2 [+43.5, +55.3] | -  |
| 2 | top1 | 1252 | 0.24681 | label:R.mention | 40.6134 | +0.7 [+0.6, +0.8] | +40.3 [+33.7, +47.9] | -0.1 [-0.2, -0.1] BELOW |
| 2 | top1 | 1252 | 0.24681 | label:R.name_comention | 18.9393 | +2.0 [+1.8, +2.1] | +116.9 [+107.3, +127.3] | +1.1 [+1.0, +1.3] ABOVE |
| 2 | top1 | 1252 | 0.24681 | label:R.title_comention | 0.8435 | +5.7 [+4.2, +7.4] | +337.3 [+251.2, +435.0] | +4.9 [+3.4, +6.5] ABOVE |
| 2 | top1 | 1252 | 0.24681 | label:K.knn | 3.9537 | +3.2 [+2.7, +3.7] | +189.5 [+162.5, +216.5] | +2.4 [+1.9, +2.8] ABOVE |
| 3 | found | 661 | 0.05144 | any | 53.0452 | +0.6 [+0.5, +0.7] | +21.9 [+19.2, +24.8] | -  |
| 3 | found | 661 | 0.05144 | label:R.mention | 36.7631 | +0.5 [+0.4, +0.6] | +16.8 [+13.6, +20.1] | -0.1 [-0.2, -0.1] BELOW |
| 3 | found | 661 | 0.05144 | label:R.name_comention | 15.9438 | +1.2 [+1.0, +1.3] | +41.9 [+36.0, +48.1] | +0.6 [+0.4, +0.7] ABOVE |
| 3 | found | 661 | 0.05144 | label:R.title_comention | 0.2821 | +8.2 [+4.9, +12.3] | +292.7 [+175.3, +436.1] | +7.6 [+4.3, +11.6] ABOVE |
| 3 | found | 661 | 0.05144 | label:K.knn | 4.1396 | +2.0 [+1.6, +2.4] | +72.0 [+57.2, +87.0] | +1.4 [+1.0, +1.8] ABOVE |
| 3 | top1 | 760 | 0.04474 | any | 67.6171 | +0.6 [+0.5, +0.7] | +20.1 [+17.4, +23.1] | -  |
| 3 | top1 | 760 | 0.04474 | label:R.mention | 50.2276 | +0.4 [+0.3, +0.5] | +13.6 [+11.1, +16.4] | -0.2 [-0.3, -0.2] BELOW |
| 3 | top1 | 760 | 0.04474 | label:R.name_comention | 17.7671 | +1.3 [+1.1, +1.5] | +42.4 [+36.7, +48.6] | +0.7 [+0.5, +0.9] ABOVE |
| 3 | top1 | 760 | 0.04474 | label:R.title_comention | 0.3684 | +10.7 [+7.4, +15.1] | +342.8 [+235.4, +486.7] | +10.1 [+6.8, +14.4] ABOVE |
| 3 | top1 | 760 | 0.04474 | label:K.knn | 4.2474 | +3.1 [+2.5, +3.7] | +99.1 [+81.3, +117.6] | +2.5 [+1.9, +3.0] ABOVE |
| 4 | found | 324 | 0.0 | any | 55.8621 | +0.8 [+0.6, +1.0] | +18.0 [+14.7, +22.0] | -  |
| 4 | found | 324 | 0.0 | label:R.mention | 38.9126 | +0.6 [+0.5, +0.8] | +14.6 [+10.9, +19.4] | -0.1 [-0.2, -0.1] BELOW |
| 4 | found | 324 | 0.0 | label:R.name_comention | 16.3172 | +1.7 [+1.5, +2.0] | +39.6 [+33.3, +45.9] | +0.9 [+0.7, +1.1] ABOVE |
| 4 | found | 324 | 0.0 | label:R.title_comention | 0.3586 | +7.1 [+3.1, +13.5] | +162.4 [+71.0, +307.9] | +6.3 [+2.3, +12.7] ABOVE |
| 4 | found | 324 | 0.0 | label:K.knn | 4.1057 | +1.9 [+1.3, +2.5] | +43.8 [+30.1, +59.2] | +1.1 [+0.5, +1.7] ABOVE |
| 4 | top1 | 405 | 0.0 | any | 75.4691 | +0.5 [+0.4, +0.6] | +10.7 [+8.5, +13.2] | -  |
| 4 | top1 | 405 | 0.0 | label:R.mention | 57.7111 | +0.4 [+0.3, +0.5] | +8.4 [+6.3, +10.8] | -0.1 [-0.2, -0.1] BELOW |
| 4 | top1 | 405 | 0.0 | label:R.name_comention | 17.3481 | +1.1 [+0.8, +1.3] | +22.6 [+17.5, +28.0] | +0.6 [+0.4, +0.8] ABOVE |
| 4 | top1 | 405 | 0.0 | label:R.title_comention | 0.6395 | +5.0 [+2.4, +8.3] | +104.9 [+51.5, +176.0] | +4.5 [+2.0, +7.8] ABOVE |
| 4 | top1 | 405 | 0.0 | label:K.knn | 4.2938 | +2.0 [+1.4, +2.6] | +40.9 [+28.0, +54.6] | +1.4 [+0.9, +2.1] ABOVE |

| type | kind | label pair | difference | call |
|---|---|---|---|---|
| 2 | found | R.mention - R.name_comention | -1.2 [-1.4, -1.1] | BELOW |
| 2 | found | R.mention - R.title_comention | -3.3 [-4.7, -2.1] | BELOW |
| 2 | found | R.mention - K.knn | -1.8 [-2.2, -1.4] | BELOW |
| 2 | found | R.name_comention - R.title_comention | -2.1 [-3.5, -0.9] | BELOW |
| 2 | found | R.name_comention - K.knn | -0.6 [-1.0, -0.2] | BELOW |
| 2 | found | R.title_comention - K.knn | +1.5 [+0.3, +2.9] | ABOVE |
| 2 | top1 | R.mention - R.name_comention | -1.3 [-1.5, -1.1] | BELOW |
| 2 | top1 | R.mention - R.title_comention | -5.0 [-6.7, -3.5] | BELOW |
| 2 | top1 | R.mention - K.knn | -2.5 [-3.0, -2.1] | BELOW |
| 2 | top1 | R.name_comention - R.title_comention | -3.7 [-5.4, -2.2] | BELOW |
| 2 | top1 | R.name_comention - K.knn | -1.2 [-1.6, -0.8] | BELOW |
| 2 | top1 | R.title_comention - K.knn | +2.5 [+1.0, +4.2] | ABOVE |
| 3 | found | R.mention - R.name_comention | -0.7 [-0.9, -0.5] | BELOW |
| 3 | found | R.mention - R.title_comention | -7.7 [-11.8, -4.4] | BELOW |
| 3 | found | R.mention - K.knn | -1.5 [-2.0, -1.2] | BELOW |
| 3 | found | R.name_comention - R.title_comention | -7.0 [-11.1, -3.7] | BELOW |
| 3 | found | R.name_comention - K.knn | -0.8 [-1.2, -0.4] | BELOW |
| 3 | found | R.title_comention - K.knn | +6.2 [+3.0, +10.1] | ABOVE |
| 3 | top1 | R.mention - R.name_comention | -0.9 [-1.1, -0.7] | BELOW |
| 3 | top1 | R.mention - R.title_comention | -10.3 [-14.6, -7.0] | BELOW |
| 3 | top1 | R.mention - K.knn | -2.7 [-3.2, -2.1] | BELOW |
| 3 | top1 | R.name_comention - R.title_comention | -9.4 [-13.8, -6.1] | BELOW |
| 3 | top1 | R.name_comention - K.knn | -1.8 [-2.3, -1.2] | BELOW |
| 3 | top1 | R.title_comention - K.knn | +7.6 [+4.4, +11.8] | ABOVE |
| 4 | found | R.mention - R.name_comention | -1.1 [-1.3, -0.8] | BELOW |
| 4 | found | R.mention - R.title_comention | -6.4 [-12.8, -2.4] | BELOW |
| 4 | found | R.mention - K.knn | -1.3 [-1.9, -0.7] | BELOW |
| 4 | found | R.name_comention - R.title_comention | -5.3 [-11.7, -1.3] | BELOW |
| 4 | found | R.name_comention - K.knn | -0.2 [-0.8, +0.4] | LEVEL |
| 4 | found | R.title_comention - K.knn | +5.1 [+1.1, +11.6] | ABOVE |
| 4 | top1 | R.mention - R.name_comention | -0.7 [-0.9, -0.4] | BELOW |
| 4 | top1 | R.mention - R.title_comention | -4.6 [-7.9, -2.1] | BELOW |
| 4 | top1 | R.mention - K.knn | -1.6 [-2.2, -1.0] | BELOW |
| 4 | top1 | R.name_comention - R.title_comention | -3.9 [-7.3, -1.4] | BELOW |
| 4 | top1 | R.name_comention - K.knn | -0.9 [-1.6, -0.3] | BELOW |
| 4 | top1 | R.title_comention - K.knn | +3.1 [+0.6, +6.3] | ABOVE |

## D83E metaqa (s1eval): {'1': 2498, '2': 3718, '3': 3569}

top1: FLIP [{'pair': 'S - K.knn', 'ABOVE_in': ['1', '3'], 'BELOW_in': ['2']}, {'pair': 'S - R.name_comention', 'ABOVE_in': ['1', '3'], 'BELOW_in': ['2']}] | found: NO_FLIP []

| type | kind | questions | all golds in top 5 | unit | pairs/anchor | precision | lift | vs any |
|---|---|---|---|---|---|---|---|---|
| 1 | found | 33 | 0.33333 | any | 13.0571 | +0.2 [+0.0, +0.7] | +4.3 [+0.0, +15.0] | -  |
| 1 | found | 33 | 0.33333 | label:S | 9.9714 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -0.2 [-0.7, +0.0] LEVEL |
| 1 | found | 33 | 0.33333 | label:R.name_comention | 0.1429 | +20.0 [+0.0, +50.0] | +391.4 [+0.0, +1088.9] | +19.8 [+0.0, +49.8] LEVEL |
| 1 | found | 33 | 0.33333 | label:K.knn | 3.3429 | +0.9 [+0.0, +2.8] | +16.7 [+0.0, +58.1] | +0.6 [+0.0, +2.1] LEVEL |
| 1 | top1 | 2498 | 0.00441 | any | 10.5184 | +16.4 [+15.6, +17.2] | +391.9 [+380.5, +403.0] | -  |
| 1 | top1 | 2498 | 0.00441 | label:S | 6.7942 | +25.3 [+24.2, +26.5] | +606.4 [+585.8, +626.1] | +9.0 [+8.4, +9.5] ABOVE |
| 1 | top1 | 2498 | 0.00441 | label:R.name_comention | 0.7574 | +0.1 [+0.0, +0.2] | +1.3 [+0.0, +4.3] | -16.3 [-17.1, -15.5] BELOW |
| 1 | top1 | 2498 | 0.00441 | label:K.knn | 3.0797 | +0.0 [+0.0, +0.1] | +0.9 [+0.0, +2.2] | -16.3 [-17.1, -15.6] BELOW |
| 2 | found | 79 | 0.1519 | any | 15.2626 | +2.1 [+0.9, +3.4] | +7.2 [+3.0, +13.1] | -  |
| 2 | found | 79 | 0.1519 | label:S | 12.3535 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -2.1 [-3.4, -0.9] BELOW |
| 2 | found | 79 | 0.1519 | label:R.name_comention | 0.3535 | +8.6 [+0.0, +30.0] | +30.1 [+0.0, +106.2] | +6.5 [-1.9, +27.4] LEVEL |
| 2 | found | 79 | 0.1519 | label:K.knn | 2.6566 | +11.4 [+5.0, +18.1] | +40.1 [+17.2, +70.5] | +9.4 [+4.0, +15.0] ABOVE |
| 2 | top1 | 3718 | 0.00323 | any | 7.9387 | +0.2 [+0.2, +0.3] | +1.7 [+1.3, +2.3] | -  |
| 2 | top1 | 3718 | 0.00323 | label:S | 4.4172 | +0.0 [+0.0, +0.1] | +0.4 [+0.1, +0.8] | -0.2 [-0.3, -0.1] BELOW |
| 2 | top1 | 3718 | 0.00323 | label:R.name_comention | 0.2009 | +0.7 [+0.1, +1.5] | +4.9 [+0.8, +10.9] | +0.4 [-0.1, +1.2] LEVEL |
| 2 | top1 | 3718 | 0.00323 | label:K.knn | 3.355 | +0.5 [+0.3, +0.6] | +3.4 [+2.4, +4.5] | +0.2 [+0.1, +0.3] ABOVE |
| 3 | found | 22 | 0.09091 | any | 19.1364 | +0.5 [+0.0, +1.3] | +1.0 [+0.0, +3.1] | -  |
| 3 | found | 22 | 0.09091 | label:S | 14.5909 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -0.5 [-1.3, +0.0] LEVEL |
| 3 | found | 22 | 0.09091 | label:K.knn | 4.6364 | +2.0 [+0.0, +4.8] | +4.0 [+0.0, +12.0] | +1.5 [+0.0, +3.7] LEVEL |
| 3 | top1 | 3569 | 0.00056 | any | 11.9199 | +3.7 [+3.5, +3.9] | +12.7 [+12.0, +13.5] | -  |
| 3 | top1 | 3569 | 0.00056 | label:S | 8.1533 | +5.4 [+5.1, +5.6] | +18.6 [+17.5, +19.7] | +1.7 [+1.6, +1.8] ABOVE |
| 3 | top1 | 3569 | 0.00056 | label:R.name_comention | 0.9204 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -3.7 [-3.9, -3.5] BELOW |
| 3 | top1 | 3569 | 0.00056 | label:K.knn | 3.021 | +0.0 [+0.0, +0.0] | +0.1 [+0.0, +0.2] | -3.7 [-3.8, -3.5] BELOW |

| type | kind | label pair | difference | call |
|---|---|---|---|---|
| 1 | found | S - R.name_comention | -20.0 [-50.0, +0.0] | LEVEL |
| 1 | found | S - K.knn | -0.9 [-2.8, +0.0] | LEVEL |
| 1 | found | R.name_comention - K.knn | +19.1 [+0.0, +49.1] | LEVEL |
| 1 | top1 | S - R.name_comention | +25.3 [+24.1, +26.4] | ABOVE |
| 1 | top1 | S - K.knn | +25.3 [+24.2, +26.4] | ABOVE |
| 1 | top1 | R.name_comention - K.knn | +0.0 [-0.1, +0.1] | LEVEL |
| 2 | found | S - R.name_comention | -8.6 [-30.0, +0.0] | LEVEL |
| 2 | found | S - K.knn | -11.4 [-18.1, -5.0] | BELOW |
| 2 | found | R.name_comention - K.knn | -2.8 [-11.7, +16.5] | LEVEL |
| 2 | top1 | S - R.name_comention | -0.6 [-1.4, -0.1] | BELOW |
| 2 | top1 | S - K.knn | -0.4 [-0.6, -0.3] | BELOW |
| 2 | top1 | R.name_comention - K.knn | +0.2 [-0.4, +1.0] | LEVEL |
| 3 | found | S - K.knn | -2.0 [-4.8, +0.0] | LEVEL |
| 3 | top1 | S - R.name_comention | +5.4 [+5.1, +5.6] | ABOVE |
| 3 | top1 | S - K.knn | +5.4 [+5.1, +5.6] | ABOVE |
| 3 | top1 | R.name_comention - K.knn | -0.0 [-0.0, +0.0] | LEVEL |

## B1V hotpotqa (b1): {'bridge': 811, 'comparison': 189}

top1: NO_FLIP [] | found: NO_FLIP []

| type | kind | questions | all golds in top 5 | unit | pairs/anchor | precision | lift | vs any |
|---|---|---|---|---|---|---|---|---|
| bridge | found | 799 | 0.70713 | any | 5.8096 | +2.9 [+2.5, +3.3] | +5.9 [+5.5, +6.3] | -  |
| bridge | found | 799 | 0.70713 | label:R.mention | 0.59 | +20.1 [+16.9, +23.7] | +41.1 [+35.8, +47.1] | +17.2 [+14.2, +20.6] ABOVE |
| bridge | found | 799 | 0.70713 | label:R.name_comention | 4.2931 | +2.8 [+2.4, +3.3] | +5.8 [+5.2, +6.4] | -0.1 [-0.3, +0.2] LEVEL |
| bridge | found | 799 | 0.70713 | label:R.title_comention | 0.1469 | +20.4 [+14.4, +28.2] | +41.8 [+29.5, +56.8] | +17.5 [+11.6, +25.1] ABOVE |
| bridge | found | 799 | 0.70713 | label:K.knn | 2.1807 | +4.4 [+3.7, +5.2] | +9.0 [+7.9, +10.2] | +1.5 [+1.0, +2.1] ABOVE |
| bridge | top1 | 811 | 0.69667 | any | 5.9951 | +4.4 [+3.9, +5.0] | +8.4 [+7.8, +9.1] | -  |
| bridge | top1 | 811 | 0.69667 | label:R.mention | 0.5586 | +31.8 [+27.1, +36.7] | +60.1 [+51.8, +70.4] | +27.3 [+23.0, +32.1] ABOVE |
| bridge | top1 | 811 | 0.69667 | label:R.name_comention | 4.4451 | +4.4 [+3.8, +5.1] | +8.4 [+7.5, +9.4] | -0.0 [-0.3, +0.4] LEVEL |
| bridge | top1 | 811 | 0.69667 | label:R.title_comention | 0.1517 | +26.0 [+17.5, +37.2] | +49.2 [+33.3, +69.3] | +21.6 [+13.1, +32.3] ABOVE |
| bridge | top1 | 811 | 0.69667 | label:K.knn | 2.2429 | +6.7 [+5.6, +7.8] | +12.7 [+11.0, +14.5] | +2.3 [+1.5, +3.1] ABOVE |
| comparison | found | 189 | 0.9418 | any | 5.9046 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -  |
| comparison | found | 189 | 0.9418 | label:R.mention | 1.4441 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | found | 189 | 0.9418 | label:R.name_comention | 3.8774 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | found | 189 | 0.9418 | label:R.title_comention | 0.267 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | found | 189 | 0.9418 | label:K.knn | 2.7193 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | top1 | 189 | 0.9418 | any | 6.3598 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -  |
| comparison | top1 | 189 | 0.9418 | label:R.mention | 1.619 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | top1 | 189 | 0.9418 | label:R.name_comention | 4.5397 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | top1 | 189 | 0.9418 | label:R.title_comention | 0.2011 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | top1 | 189 | 0.9418 | label:K.knn | 2.7196 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |

| type | kind | label pair | difference | call |
|---|---|---|---|---|
| bridge | found | R.mention - R.name_comention | +17.3 [+14.3, +20.7] | ABOVE |
| bridge | found | R.mention - R.title_comention | -0.3 [-7.6, +5.7] | LEVEL |
| bridge | found | R.mention - K.knn | +15.7 [+12.8, +19.1] | ABOVE |
| bridge | found | R.name_comention - R.title_comention | -17.6 [-25.3, -11.6] | BELOW |
| bridge | found | R.name_comention - K.knn | -1.6 [-2.2, -1.0] | BELOW |
| bridge | found | R.title_comention - K.knn | +16.0 [+10.0, +23.6] | ABOVE |
| bridge | top1 | R.mention - R.name_comention | +27.4 [+22.9, +32.0] | ABOVE |
| bridge | top1 | R.mention - R.title_comention | +5.8 [-5.1, +14.5] | LEVEL |
| bridge | top1 | R.mention - K.knn | +25.1 [+20.7, +29.8] | ABOVE |
| bridge | top1 | R.name_comention - R.title_comention | -21.6 [-32.6, -13.1] | BELOW |
| bridge | top1 | R.name_comention - K.knn | -2.3 [-3.1, -1.3] | BELOW |
| bridge | top1 | R.title_comention - K.knn | +19.3 [+10.8, +30.3] | ABOVE |
| comparison | found | R.mention - R.name_comention | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | found | R.mention - R.title_comention | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | found | R.mention - K.knn | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | found | R.name_comention - R.title_comention | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | found | R.name_comention - K.knn | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | found | R.title_comention - K.knn | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | top1 | R.mention - R.name_comention | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | top1 | R.mention - R.title_comention | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | top1 | R.mention - K.knn | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | top1 | R.name_comention - R.title_comention | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | top1 | R.name_comention - K.knn | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | top1 | R.title_comention - K.knn | +0.0 [+0.0, +0.0] | LEVEL |

## B1V 2wiki (b1): {'compositional': 413, 'comparison': 244, 'bridge_comparison': 235, 'inference': 108}

top1: FLIP [{'pair': 'R.name_comention - K.knn', 'ABOVE_in': ['bridge_comparison'], 'BELOW_in': ['compositional', 'inference']}] | found: FLIP [{'pair': 'R.name_comention - K.knn', 'ABOVE_in': ['bridge_comparison'], 'BELOW_in': ['compositional', 'inference']}]

| type | kind | questions | all golds in top 5 | unit | pairs/anchor | precision | lift | vs any |
|---|---|---|---|---|---|---|---|---|
| compositional | found | 412 | 0.3665 | any | 5.9316 | +8.1 [+7.2, +9.2] | +11.2 [+10.2, +12.3] | -  |
| compositional | found | 412 | 0.3665 | label:R.mention | 0.6026 | +70.2 [+64.5, +76.1] | +97.3 [+88.2, +106.7] | +62.1 [+56.8, +67.7] ABOVE |
| compositional | found | 412 | 0.3665 | label:R.name_comention | 4.1793 | +9.2 [+7.9, +10.7] | +12.7 [+11.2, +14.6] | +1.1 [+0.5, +1.8] ABOVE |
| compositional | found | 412 | 0.3665 | label:R.title_comention | 0.1201 | +64.6 [+50.8, +78.7] | +89.5 [+70.3, +109.0] | +56.5 [+42.9, +70.1] ABOVE |
| compositional | found | 412 | 0.3665 | label:K.knn | 2.5416 | +11.8 [+10.3, +13.3] | +16.3 [+14.7, +18.0] | +3.7 [+2.4, +4.9] ABOVE |
| compositional | top1 | 413 | 0.36562 | any | 5.9031 | +10.4 [+9.4, +11.4] | +14.3 [+13.1, +15.5] | -  |
| compositional | top1 | 413 | 0.36562 | label:R.mention | 0.7046 | +75.6 [+70.5, +80.7] | +104.0 [+95.3, +114.3] | +65.2 [+60.4, +70.2] ABOVE |
| compositional | top1 | 413 | 0.36562 | label:R.name_comention | 4.046 | +12.0 [+10.5, +13.7] | +16.5 [+14.5, +18.7] | +1.6 [+0.8, +2.5] ABOVE |
| compositional | top1 | 413 | 0.36562 | label:R.title_comention | 0.1356 | +73.2 [+60.4, +85.7] | +100.7 [+82.6, +119.1] | +62.8 [+50.4, +75.2] ABOVE |
| compositional | top1 | 413 | 0.36562 | label:K.knn | 2.707 | +14.3 [+12.6, +16.0] | +19.7 [+17.8, +21.7] | +3.9 [+2.4, +5.4] ABOVE |
| comparison | found | 244 | 0.95492 | any | 3.2704 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -  |
| comparison | found | 244 | 0.95492 | label:R.mention | 0.1468 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | found | 244 | 0.95492 | label:R.name_comention | 2.0 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | found | 244 | 0.95492 | label:R.title_comention | 0.0377 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | found | 244 | 0.95492 | label:K.knn | 1.6101 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | top1 | 244 | 0.95492 | any | 3.6475 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -  |
| comparison | top1 | 244 | 0.95492 | label:R.mention | 0.1557 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | top1 | 244 | 0.95492 | label:R.name_comention | 2.3197 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | top1 | 244 | 0.95492 | label:R.title_comention | 0.0287 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| comparison | top1 | 244 | 0.95492 | label:K.knn | 1.7951 | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] LEVEL |
| bridge_comparison | found | 235 | 0.09362 | any | 4.9439 | +14.9 [+13.7, +16.1] | +10.1 [+9.4, +10.9] | -  |
| bridge_comparison | found | 235 | 0.09362 | label:R.mention | 0.8697 | +79.0 [+74.7, +83.8] | +53.9 [+50.5, +57.6] | +64.2 [+60.4, +68.2] ABOVE |
| bridge_comparison | found | 235 | 0.09362 | label:R.name_comention | 3.2445 | +18.7 [+16.7, +20.9] | +12.8 [+11.5, +14.2] | +3.8 [+2.8, +5.1] ABOVE |
| bridge_comparison | found | 235 | 0.09362 | label:R.title_comention | 0.1483 | +81.1 [+69.4, +91.5] | +55.3 [+47.8, +62.7] | +66.2 [+55.2, +76.3] ABOVE |
| bridge_comparison | found | 235 | 0.09362 | label:K.knn | 2.3968 | +13.4 [+11.6, +15.2] | +9.1 [+8.0, +10.3] | -1.5 [-3.1, +0.1] LEVEL |
| bridge_comparison | top1 | 235 | 0.09362 | any | 5.1319 | +15.0 [+13.4, +16.6] | +10.2 [+9.2, +11.3] | -  |
| bridge_comparison | top1 | 235 | 0.09362 | label:R.mention | 0.8809 | +80.7 [+74.8, +86.8] | +55.0 [+50.4, +60.2] | +65.7 [+60.2, +71.3] ABOVE |
| bridge_comparison | top1 | 235 | 0.09362 | label:R.name_comention | 3.3404 | +19.0 [+16.2, +22.1] | +12.9 [+11.0, +15.1] | +4.0 [+2.4, +5.8] ABOVE |
| bridge_comparison | top1 | 235 | 0.09362 | label:R.title_comention | 0.1574 | +83.8 [+71.4, +94.6] | +57.1 [+48.7, +65.4] | +68.8 [+56.7, +79.3] ABOVE |
| bridge_comparison | top1 | 235 | 0.09362 | label:K.knn | 2.4298 | +15.4 [+12.9, +18.0] | +10.5 [+8.8, +12.2] | +0.4 [-1.9, +2.8] LEVEL |
| inference | found | 108 | 0.5 | any | 7.3949 | +4.7 [+3.6, +5.9] | +5.8 [+5.0, +6.9] | -  |
| inference | found | 108 | 0.5 | label:R.mention | 0.4395 | +63.8 [+51.3, +77.1] | +79.9 [+65.3, +98.8] | +59.1 [+47.2, +71.9] ABOVE |
| inference | found | 108 | 0.5 | label:R.name_comention | 6.0255 | +4.7 [+3.4, +6.1] | +5.8 [+4.8, +7.1] | +0.0 [-0.6, +0.5] LEVEL |
| inference | found | 108 | 0.5 | label:R.title_comention | 0.172 | +22.2 [+7.7, +50.0] | +27.9 [+10.1, +59.9] | +17.6 [+3.3, +44.3] ABOVE |
| inference | found | 108 | 0.5 | label:K.knn | 2.2866 | +11.7 [+8.7, +15.2] | +14.7 [+11.9, +17.5] | +7.0 [+4.8, +9.7] ABOVE |
| inference | top1 | 108 | 0.5 | any | 7.6389 | +6.2 [+4.8, +7.7] | +7.7 [+6.6, +9.1] | -  |
| inference | top1 | 108 | 0.5 | label:R.mention | 0.537 | +70.7 [+59.1, +82.7] | +88.6 [+72.5, +111.4] | +64.5 [+53.5, +76.5] ABOVE |
| inference | top1 | 108 | 0.5 | label:R.name_comention | 6.25 | +6.1 [+4.6, +7.8] | +7.6 [+6.1, +9.3] | -0.1 [-0.9, +0.6] LEVEL |
| inference | top1 | 108 | 0.5 | label:R.title_comention | 0.1389 | +40.0 [+16.7, +71.4] | +50.1 [+21.3, +90.4] | +33.8 [+10.5, +64.7] ABOVE |
| inference | top1 | 108 | 0.5 | label:K.knn | 2.4815 | +15.7 [+12.2, +19.5] | +19.6 [+16.2, +23.3] | +9.5 [+6.7, +12.5] ABOVE |

| type | kind | label pair | difference | call |
|---|---|---|---|---|
| compositional | found | R.mention - R.name_comention | +61.0 [+55.8, +66.5] | ABOVE |
| compositional | found | R.mention - R.title_comention | +5.6 [-8.2, +19.4] | LEVEL |
| compositional | found | R.mention - K.knn | +58.5 [+52.9, +64.0] | ABOVE |
| compositional | found | R.name_comention - R.title_comention | -55.4 [-69.2, -41.5] | BELOW |
| compositional | found | R.name_comention - K.knn | -2.6 [-4.0, -1.0] | BELOW |
| compositional | found | R.title_comention - K.knn | +52.8 [+39.0, +66.5] | ABOVE |
| compositional | top1 | R.mention - R.name_comention | +63.6 [+58.8, +68.7] | ABOVE |
| compositional | top1 | R.mention - R.title_comention | +2.4 [-9.9, +16.0] | LEVEL |
| compositional | top1 | R.mention - K.knn | +61.3 [+56.5, +66.1] | ABOVE |
| compositional | top1 | R.name_comention - R.title_comention | -61.2 [-73.5, -48.3] | BELOW |
| compositional | top1 | R.name_comention - K.knn | -2.3 [-4.2, -0.4] | BELOW |
| compositional | top1 | R.title_comention - K.knn | +58.9 [+46.2, +71.5] | ABOVE |
| comparison | found | R.mention - R.name_comention | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | found | R.mention - R.title_comention | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | found | R.mention - K.knn | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | found | R.name_comention - R.title_comention | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | found | R.name_comention - K.knn | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | found | R.title_comention - K.knn | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | top1 | R.mention - R.name_comention | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | top1 | R.mention - R.title_comention | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | top1 | R.mention - K.knn | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | top1 | R.name_comention - R.title_comention | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | top1 | R.name_comention - K.knn | +0.0 [+0.0, +0.0] | LEVEL |
| comparison | top1 | R.title_comention - K.knn | +0.0 [+0.0, +0.0] | LEVEL |
| bridge_comparison | found | R.mention - R.name_comention | +60.3 [+56.5, +64.6] | ABOVE |
| bridge_comparison | found | R.mention - R.title_comention | -2.0 [-11.7, +8.3] | LEVEL |
| bridge_comparison | found | R.mention - K.knn | +65.7 [+61.3, +70.2] | ABOVE |
| bridge_comparison | found | R.name_comention - R.title_comention | -62.4 [-72.7, -51.1] | BELOW |
| bridge_comparison | found | R.name_comention - K.knn | +5.3 [+3.2, +7.5] | ABOVE |
| bridge_comparison | found | R.title_comention - K.knn | +67.7 [+56.1, +78.2] | ABOVE |
| bridge_comparison | top1 | R.mention - R.name_comention | +61.7 [+56.2, +67.4] | ABOVE |
| bridge_comparison | top1 | R.mention - R.title_comention | -3.1 [-13.8, +8.8] | LEVEL |
| bridge_comparison | top1 | R.mention - K.knn | +65.3 [+59.5, +71.4] | ABOVE |
| bridge_comparison | top1 | R.name_comention - R.title_comention | -64.8 [-75.4, -52.3] | BELOW |
| bridge_comparison | top1 | R.name_comention - K.knn | +3.6 [+0.4, +7.0] | ABOVE |
| bridge_comparison | top1 | R.title_comention - K.knn | +68.4 [+56.2, +79.5] | ABOVE |
| inference | found | R.mention - R.name_comention | +59.1 [+47.1, +71.9] | ABOVE |
| inference | found | R.mention - R.title_comention | +41.5 [+17.6, +57.9] | ABOVE |
| inference | found | R.mention - K.knn | +52.1 [+40.9, +64.1] | ABOVE |
| inference | found | R.name_comention - R.title_comention | -17.6 [-44.5, -3.1] | BELOW |
| inference | found | R.name_comention - K.knn | -7.0 [-9.8, -4.6] | BELOW |
| inference | found | R.title_comention - K.knn | +10.5 [-3.5, +36.8] | LEVEL |
| inference | top1 | R.mention - R.name_comention | +64.6 [+53.3, +76.7] | ABOVE |
| inference | top1 | R.mention - R.title_comention | +30.7 [-1.0, +55.8] | LEVEL |
| inference | top1 | R.mention - K.knn | +55.0 [+44.7, +66.1] | ABOVE |
| inference | top1 | R.name_comention - R.title_comention | -33.9 [-64.7, -10.1] | BELOW |
| inference | top1 | R.name_comention - K.knn | -9.6 [-12.8, -6.7] | BELOW |
| inference | top1 | R.title_comention - K.knn | +24.3 [+1.8, +54.7] | ABOVE |

## B1V musique (b1): {'2': 518, '3': 316, '4': 166}

top1: NO_FLIP [] | found: NO_FLIP []

| type | kind | questions | all golds in top 5 | unit | pairs/anchor | precision | lift | vs any |
|---|---|---|---|---|---|---|---|---|
| 2 | found | 486 | 0.39506 | any | 23.2618 | +1.6 [+1.4, +1.8] | +89.0 [+80.8, +98.2] | -  |
| 2 | found | 486 | 0.39506 | label:R.mention | 6.7337 | +2.4 [+1.8, +3.1] | +133.9 [+106.1, +168.7] | +0.8 [+0.4, +1.3] ABOVE |
| 2 | found | 486 | 0.39506 | label:R.name_comention | 15.7559 | +2.0 [+1.7, +2.3] | +111.2 [+99.7, +124.1] | +0.4 [+0.3, +0.5] ABOVE |
| 2 | found | 486 | 0.39506 | label:R.title_comention | 0.8861 | +3.3 [+2.0, +5.1] | +188.4 [+113.2, +288.4] | +1.8 [+0.4, +3.5] ABOVE |
| 2 | found | 486 | 0.39506 | label:K.knn | 3.9068 | +3.4 [+2.7, +4.1] | +190.1 [+155.5, +224.8] | +1.8 [+1.2, +2.4] ABOVE |
| 2 | top1 | 518 | 0.37066 | any | 23.4672 | +2.0 [+1.8, +2.4] | +101.1 [+89.3, +114.9] | -  |
| 2 | top1 | 518 | 0.37066 | label:R.mention | 7.5598 | +2.9 [+2.3, +3.9] | +145.0 [+110.8, +191.9] | +0.9 [+0.4, +1.6] ABOVE |
| 2 | top1 | 518 | 0.37066 | label:R.name_comention | 15.2915 | +2.6 [+2.3, +3.0] | +128.4 [+113.1, +144.9] | +0.6 [+0.4, +0.8] ABOVE |
| 2 | top1 | 518 | 0.37066 | label:R.title_comention | 0.9923 | +4.5 [+2.7, +6.7] | +220.9 [+133.8, +331.2] | +2.4 [+0.7, +4.7] ABOVE |
| 2 | top1 | 518 | 0.37066 | label:K.knn | 3.7973 | +4.8 [+3.9, +5.7] | +235.9 [+193.2, +279.6] | +2.7 [+1.9, +3.6] ABOVE |
| 3 | found | 294 | 0.07823 | any | 31.4331 | +1.8 [+1.5, +2.0] | +46.6 [+41.0, +53.2] | -  |
| 3 | found | 294 | 0.07823 | label:R.mention | 10.6527 | +1.9 [+1.4, +2.5] | +49.7 [+37.7, +65.1] | +0.1 [-0.2, +0.5] LEVEL |
| 3 | found | 294 | 0.07823 | label:R.name_comention | 20.1337 | +2.1 [+1.8, +2.4] | +55.2 [+48.5, +62.2] | +0.3 [+0.2, +0.5] ABOVE |
| 3 | found | 294 | 0.07823 | label:R.title_comention | 1.9381 | +2.6 [+1.7, +3.5] | +68.3 [+44.9, +93.9] | +0.8 [-0.1, +1.8] LEVEL |
| 3 | found | 294 | 0.07823 | label:K.knn | 3.996 | +3.5 [+2.8, +4.2] | +92.7 [+75.1, +110.7] | +1.7 [+1.1, +2.4] ABOVE |
| 3 | top1 | 316 | 0.07278 | any | 35.8956 | +1.7 [+1.4, +2.1] | +42.4 [+34.1, +52.0] | -  |
| 3 | top1 | 316 | 0.07278 | label:R.mention | 14.5823 | +1.6 [+1.0, +2.6] | +40.1 [+24.8, +63.3] | -0.1 [-0.5, +0.5] LEVEL |
| 3 | top1 | 316 | 0.07278 | label:R.name_comention | 21.2911 | +2.0 [+1.8, +2.3] | +49.5 [+42.2, +56.9] | +0.3 [-0.0, +0.6] LEVEL |
| 3 | top1 | 316 | 0.07278 | label:R.title_comention | 1.5823 | +3.8 [+2.2, +5.8] | +92.3 [+53.4, +140.1] | +2.1 [+0.5, +3.9] ABOVE |
| 3 | top1 | 316 | 0.07278 | label:K.knn | 3.9842 | +4.4 [+3.4, +5.6] | +108.0 [+84.0, +133.1] | +2.7 [+1.7, +3.7] ABOVE |
| 4 | found | 155 | 0.0 | any | 31.3418 | +2.5 [+2.2, +2.9] | +35.3 [+30.4, +40.7] | -  |
| 4 | found | 155 | 0.0 | label:R.mention | 7.692 | +2.8 [+1.9, +4.0] | +39.3 [+26.7, +55.9] | +0.3 [-0.4, +1.3] LEVEL |
| 4 | found | 155 | 0.0 | label:R.name_comention | 23.3671 | +3.0 [+2.5, +3.4] | +41.6 [+35.6, +48.1] | +0.4 [+0.2, +0.7] ABOVE |
| 4 | found | 155 | 0.0 | label:R.title_comention | 0.9198 | +3.7 [+1.4, +6.7] | +51.5 [+20.4, +95.6] | +1.2 [-1.0, +4.2] LEVEL |
| 4 | found | 155 | 0.0 | label:K.knn | 4.1561 | +3.7 [+2.6, +4.8] | +51.3 [+36.2, +67.8] | +1.1 [+0.1, +2.2] ABOVE |
| 4 | top1 | 166 | 0.0 | any | 34.9639 | +2.4 [+2.1, +2.8] | +32.5 [+27.7, +38.1] | -  |
| 4 | top1 | 166 | 0.0 | label:R.mention | 9.8554 | +2.0 [+1.3, +3.0] | +27.2 [+17.8, +41.0] | -0.4 [-1.0, +0.4] LEVEL |
| 4 | top1 | 166 | 0.0 | label:R.name_comention | 25.7229 | +2.7 [+2.3, +3.3] | +36.9 [+30.9, +43.9] | +0.3 [+0.1, +0.6] ABOVE |
| 4 | top1 | 166 | 0.0 | label:R.title_comention | 0.9699 | +3.7 [+1.1, +7.0] | +50.2 [+14.7, +95.6] | +1.3 [-1.4, +4.6] LEVEL |
| 4 | top1 | 166 | 0.0 | label:K.knn | 4.1687 | +4.2 [+2.8, +5.7] | +56.5 [+38.9, +75.6] | +1.8 [+0.6, +3.1] ABOVE |

| type | kind | label pair | difference | call |
|---|---|---|---|---|
| 2 | found | R.mention - R.name_comention | +0.4 [-0.1, +1.0] | LEVEL |
| 2 | found | R.mention - R.title_comention | -1.0 [-2.6, +0.4] | LEVEL |
| 2 | found | R.mention - K.knn | -1.0 [-1.7, -0.3] | BELOW |
| 2 | found | R.name_comention - R.title_comention | -1.4 [-3.1, +0.1] | LEVEL |
| 2 | found | R.name_comention - K.knn | -1.4 [-2.0, -0.8] | BELOW |
| 2 | found | R.title_comention - K.knn | -0.0 [-1.4, +1.7] | LEVEL |
| 2 | top1 | R.mention - R.name_comention | +0.3 [-0.3, +1.2] | LEVEL |
| 2 | top1 | R.mention - R.title_comention | -1.5 [-3.7, +0.4] | LEVEL |
| 2 | top1 | R.mention - K.knn | -1.8 [-2.9, -0.7] | BELOW |
| 2 | top1 | R.name_comention - R.title_comention | -1.9 [-4.2, -0.0] | BELOW |
| 2 | top1 | R.name_comention - K.knn | -2.2 [-3.0, -1.4] | BELOW |
| 2 | top1 | R.title_comention - K.knn | -0.3 [-2.3, +1.9] | LEVEL |
| 3 | found | R.mention - R.name_comention | -0.2 [-0.6, +0.3] | LEVEL |
| 3 | found | R.mention - R.title_comention | -0.7 [-1.7, +0.3] | LEVEL |
| 3 | found | R.mention - K.knn | -1.6 [-2.3, -0.9] | BELOW |
| 3 | found | R.name_comention - R.title_comention | -0.5 [-1.5, +0.4] | LEVEL |
| 3 | found | R.name_comention - K.knn | -1.4 [-2.1, -0.7] | BELOW |
| 3 | found | R.title_comention - K.knn | -0.9 [-2.0, +0.2] | LEVEL |
| 3 | top1 | R.mention - R.name_comention | -0.4 [-1.0, +0.6] | LEVEL |
| 3 | top1 | R.mention - R.title_comention | -2.2 [-4.2, -0.4] | BELOW |
| 3 | top1 | R.mention - K.knn | -2.8 [-4.0, -1.6] | BELOW |
| 3 | top1 | R.name_comention - R.title_comention | -1.8 [-3.8, -0.1] | BELOW |
| 3 | top1 | R.name_comention - K.knn | -2.4 [-3.4, -1.4] | BELOW |
| 3 | top1 | R.title_comention - K.knn | -0.6 [-2.4, +1.3] | LEVEL |
| 4 | found | R.mention - R.name_comention | -0.2 [-1.0, +0.9] | LEVEL |
| 4 | found | R.mention - R.title_comention | -0.9 [-4.2, +1.7] | LEVEL |
| 4 | found | R.mention - K.knn | -0.9 [-2.0, +0.4] | LEVEL |
| 4 | found | R.name_comention - R.title_comention | -0.7 [-3.8, +1.6] | LEVEL |
| 4 | found | R.name_comention - K.knn | -0.7 [-1.7, +0.4] | LEVEL |
| 4 | found | R.title_comention - K.knn | +0.0 [-2.5, +3.3] | LEVEL |
| 4 | top1 | R.mention - R.name_comention | -0.7 [-1.5, +0.2] | LEVEL |
| 4 | top1 | R.mention - R.title_comention | -1.7 [-5.1, +1.2] | LEVEL |
| 4 | top1 | R.mention - K.knn | -2.2 [-3.6, -0.7] | BELOW |
| 4 | top1 | R.name_comention - R.title_comention | -1.0 [-4.4, +1.9] | LEVEL |
| 4 | top1 | R.name_comention - K.knn | -1.5 [-2.9, -0.1] | BELOW |
| 4 | top1 | R.title_comention - K.knn | -0.5 [-3.5, +3.1] | LEVEL |

