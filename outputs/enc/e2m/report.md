# ENC E2M: mpnet mapped into the frozen space, first stage (report)

Written by `outputs/mp_unified/zenc2m.py report` under docs/ENC_ENCODER_COMPARISON.md section 13. R@k x100 on s1eval; a gap is the read minus E0's frozen read in points (won back: minus E0's mpnet read), paired 95% bootstrap over questions (2,000, seed 0). 'both' decides; 'qonly' (frozen documents) is beside.

G1 (the run's raw mpnet read equals E0's): {"squad": true, "musique": true}

| arm | read | carve | R@5 | gap to frozen | won back | R@20 gap | R@1000 gap | RRF@5 gap |
|---|---|---|---|---|---|---|---|---|
| ridge | both | squad | 66.98 | -20.72 [-21.54, -19.91] | -6.54 [-7.18, -5.95] | -13.88 | -1.49 | -9.21 |
| ridge | both | musique | 22.75 | -22.97 [-24.12, -21.77] | -11.08 [-11.99, -10.18] | -25.12 | -12.23 | -12.94 |
| ridge | qonly | squad | 63.93 | -23.76 [-24.54, -23.0] | -9.58 [-10.35, -8.85] | -14.12 | -1.04 | -8.98 |
| ridge | qonly | musique | 15.04 | -30.68 [-31.83, -29.54] | -18.79 [-19.89, -17.7] | -33.17 | -15.34 | -18.92 |
| mlp | both | squad | 66.98 | -20.72 [-21.54, -19.91] | -6.54 [-7.18, -5.95] | -13.88 | -1.49 | -9.21 |
| mlp | both | musique | 22.75 | -22.97 [-24.12, -21.77] | -11.08 [-11.99, -10.18] | -25.12 | -12.23 | -12.94 |
| mlp | qonly | squad | 63.93 | -23.76 [-24.54, -23.0] | -9.58 [-10.35, -8.85] | -14.12 | -1.04 | -8.98 |
| mlp | qonly | musique | 15.04 | -30.68 [-31.83, -29.54] | -18.79 [-19.89, -17.7] | -33.17 | -15.34 | -18.92 |

Bars (13.5): {"zrckpn": {"squad": -5.12, "musique": -3.11}, "gnx": {"squad": -5.41, "musique": -2.31}}

Calls: {"ridge": "STOP", "mlp": "STOP"}; stage: STOP

Fit: {"ridge": {"q": {"lambda": 0.1, "held_cos": 0.749174, "seconds": 9.0, "train": 135231, "held": 15026}, "d": {"lambda": 0.1, "held_cos": 0.725069, "seconds": 5.1, "train": 120541, "held": 13393}}, "mlp": {"q": {"best_epoch": 0, "held_cos": 0.749174, "epoch0_cos": 0.749174, "seconds": 22.3, "train": 135231, "held": 15026}, "d": {"best_epoch": 0, "held_cos": 0.725069, "epoch0_cos": 0.725069, "seconds": 17.7, "train": 120541, "held": 13393}}}
