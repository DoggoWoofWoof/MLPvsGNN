# ENC E2N: node memory on HotpotQA (report)

Written by `outputs/mp_unified/zenc2n.py report` under docs/ENC_ENCODER_COMPARISON.md section 15. R@5 x100 on HotpotQA's s1eval (7405 questions; 81.1% have a gold among the feedback golds). Won back = the read minus E0's raw read of the same encoder, points, paired 95% bootstrap (2,000, seed 0).

| encoder | form | lambda | R@5 | won back | won back, seen | won back, not seen | gap to frozen | R@1000 exact share |
|---|---|---|---|---|---|---|---|---|
| mpnet | gold | 1.0 | 55.27 | 12.65 [11.92, 13.37] | 16.46 | -3.67 | -10.5 [-11.3, -9.68] | 1.0 |
| mpnet | self | 0.25 | 42.57 | -0.05 [-0.38, 0.26] | 0.2 | -1.14 | -23.2 [-23.98, -22.46] | 1.0 |
| frozen | gold | 0.5 | 74.08 | 8.31 [7.79, 8.81] | 10.84 | -2.53 | 8.31 [7.79, 8.81] | 1.0 |
| frozen | self | 0.25 | 66.63 | 0.86 [0.61, 1.12] | 1.23 | -0.75 | 0.86 [0.61, 1.12] | 1.0 |

Calls (15.4, mpnet): {"gold": "HELPS", "self": "LEVEL"}

E0's gap, mpnet to frozen: -23.15 points. Self feedback's top 1 is a gold for {"mpnet": 0.54236, "frozen": 0.815885} of the feedback questions.
