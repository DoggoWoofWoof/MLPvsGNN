# ENC E2B: the best vectors per encoder (report)

Written by `outputs/mp_unified/zenc2b.py report` under docs/ENC_ENCODER_COMPARISON.md section 16. s1eval R@k x100. Won back = the read minus E0's raw read of the arm's encoder; gap = minus E0's frozen read; points, paired 95% bootstrap over questions (2,000, seed 0). Settings chosen on s1sel. V = one vector per question and per document (q-map, fuse, prf); F = V's parts plus the windows.

- mpnet: V {"gamma": 1.0, "beta": 0.0, "W": null, "w": 0.5, "prf": null}; F {"gamma": 1.0, "beta": 0.5, "W": 64, "w": 0.5, "prf": null}
- qwen3e06: V {"gamma": 0.5, "beta": 0.0, "W": null, "w": 0.0, "prf": null}; F {"gamma": 1.0, "beta": 0.75, "W": 64, "w": 0.0, "prf": null}
- frozen: V {"gamma": 0.5, "beta": 0.0, "W": null, "w": 0.0, "prf": null}; F {"gamma": 1.0, "beta": 0.5, "W": 64, "w": 0.0, "prf": null}

| arm | recipe | carve | R@5 | won back | gap to frozen | gap to frozen's same recipe | R@20 gap | R@1000 gap | RRF@5 gap |
|---|---|---|---|---|---|---|---|---|---|
| mpnet | V | squad | 75.43 | 1.91 [1.41, 2.44] | -12.26 [-12.93, -11.61] | -12.31 [-12.97, -11.66] | -7.73 | -0.9 | -5.96 |
| mpnet | V | musique | 39.31 | 5.48 [4.65, 6.33] | -6.41 [-7.41, -5.41] | -8.32 [-9.31, -7.33] | -8.25 | -6.18 | -3.42 |
| mpnet | F | squad | 80.48 | 6.96 [6.4, 7.57] | -7.22 [-7.84, -6.59] | -9.97 [-10.56, -9.37] | -4.74 | -0.64 | -3.83 |
| mpnet | F | musique | 40.33 | 6.5 [5.6, 7.44] | -5.39 [-6.42, -4.37] | -8.24 [-9.24, -7.22] | -7.16 | -5.44 | -2.93 |
| qwen3e06 | V | squad | 82.01 | 0.64 [0.39, 0.93] | -5.69 [-6.23, -5.12] | -5.74 [-6.27, -5.17] | -3.78 | -0.5 | -2.9 |
| qwen3e06 | V | musique | 42.5 | 1.24 [0.75, 1.73] | -3.22 [-4.15, -2.26] | -5.13 [-6.03, -4.19] | -4.54 | -3.09 | -1.63 |
| qwen3e06 | F | squad | 87.08 | 5.71 [5.18, 6.27] | -0.61 [-1.15, -0.09] | -3.37 [-3.85, -2.88] | -0.29 | -0.09 | -0.29 |
| qwen3e06 | F | musique | 43.76 | 2.5 [1.74, 3.26] | -1.95 [-2.88, -0.97] | -4.81 [-5.68, -3.93] | -4.13 | -3.51 | -0.97 |
| frozen | V | squad | 87.75 | 0.05 [-0.21, 0.3] | 0.05 [-0.21, 0.3] |  | 0.21 | 0.04 | 0.0 |
| frozen | V | musique | 47.63 | 1.91 [1.38, 2.42] | 1.91 [1.38, 2.42] |  | 1.92 | 1.02 | 1.12 |
| frozen | F | squad | 90.45 | 2.75 [2.34, 3.19] | 2.75 [2.34, 3.19] |  | 1.23 | 0.07 | 0.83 |
| frozen | F | musique | 48.57 | 2.85 [2.13, 3.58] | 2.85 [2.13, 3.58] |  | 1.81 | 0.92 | 1.14 |

Calls (16.7): {"mpnet-V": {"recipe": "IMPROVES", "gate": "PARTIAL", "by": null}, "mpnet-F": {"recipe": "IMPROVES", "gate_pointer": "PARTIAL", "by": null}, "qwen3e06-V": {"recipe": "IMPROVES", "gate": "PARTIAL", "by": null}, "qwen3e06-F": {"recipe": "IMPROVES", "gate_pointer": "PARTIAL", "by": null}, "frozen-V": {"recipe": "MIXED"}, "frozen-F": {"recipe": "IMPROVES"}}

Bars (13.5's rule, per arm): {"mpnet": {"zrckpn": {"squad": -5.12, "musique": -3.11}, "gnx": {"squad": -5.41, "musique": -2.31}}, "qwen3e06": {"zrckpn": {"squad": -3.79, "musique": -1.14}, "gnx": {"squad": -4.25, "musique": -1.25}}}

Leave one out (the recipe minus the recipe with that part set back to raw, R@5 points): {"mpnet-V__squad": {"gamma": [-1.04, [-1.52, -0.6]], "w": [2.79, [2.38, 3.22]]}, "mpnet-V__musique": {"gamma": [3.96, [3.21, 4.72]], "w": [1.02, [0.5, 1.53]]}, "mpnet-F__squad": {"gamma": [-0.61, [-1.04, -0.17]], "beta": [5.05, [4.59, 5.54]], "w": [2.35, [1.99, 2.73]]}, "mpnet-F__musique": {"gamma": [3.58, [2.81, 4.38]], "beta": [1.02, [0.56, 1.47]], "w": [1.12, [0.57, 1.64]]}, "qwen3e06-V__squad": {"gamma": [0.64, [0.39, 0.93]]}, "qwen3e06-V__musique": {"gamma": [1.24, [0.75, 1.73]]}, "qwen3e06-F__squad": {"gamma": [-0.04, [-0.39, 0.3]], "beta": [5.61, [5.11, 6.13]]}, "qwen3e06-F__musique": {"gamma": [1.28, [0.68, 1.89]], "beta": [1.65, [1.07, 2.23]]}, "frozen-V__squad": {"gamma": [0.05, [-0.21, 0.3]]}, "frozen-V__musique": {"gamma": [1.91, [1.38, 2.42]]}, "frozen-F__squad": {"gamma": [-0.28, [-0.58, 0.05]], "beta": [3.62, [3.23, 4.02]]}, "frozen-F__musique": {"gamma": [1.68, [1.0, 2.39]], "beta": [0.68, [0.19, 1.16]]}}

HotpotQA beside (V's gamma and prf; mpnet's w set to 0; memory = E2N's gold form):

| encoder | read | R@5 | won back | gap to frozen (E0) | R@1000 exact share |
|---|---|---|---|---|---|
| mpnet | V | 46.01 | 3.38 [2.88, 3.89] | -19.76 [-20.5, -19.05] | 1.0 |
| mpnet | V-mem | 53.63 | 11.0 [10.3, 11.7] | -12.15 [-12.91, -11.39] | 1.0 |
| frozen | V | 65.87 | 0.1 [-0.17, 0.36] | 0.1 [-0.17, 0.36] | 1.0 |
| frozen | V-mem | 73.09 | 7.32 [6.84, 7.8] | 7.32 [6.84, 7.8] | 1.0 |

One read per arm and recipe on s1eval; every setting was chosen on s1sel.
