# CAP: zrckpn's capacity cells, seed-0 screen against zrckpn's own three seeds

zrckpn is hidden 128 at `2e-3:1e-4:0.1:8:2`. Chosen for the full run: **None**.

| cell | hidden | config | own (twelve reads) | six-read | stop | mean delta, six | GAIN / LOSS (six) |
|---|---:|---|---|---|---|---:|---|
| e16 | 128 | 2e-3:1e-4:0.1:16:4 | NOT_ADOPTED | NO_GAIN | GO | +0.0032 | 0 / 0 |
| w256 | 256 | 2e-3:1e-4:0.1:8:2 | NOT_ADOPTED | NOT_ADOPTED | GO | -0.0089 | 0 / 3 |
| w256e16 | 256 | 2e-3:1e-4:0.1:16:4 | NOT_ADOPTED | NO_GAIN | GO | -0.0029 | 0 / 0 |

Seed-0 R@5 deltas against zrckpn (\* = a GAIN or a LOSS at the read's floor):

| split | read | zrckpn s0, s1, s2 | floor | e16 | w256 | w256e16 |
|---|---|---|---:|---:|---:|---:|
| WQ-musique | metaqa | 0.4126, 0.4256, 0.4167 | 0.0193 | -0.0385* | +0.0573* | -0.0462* |
| WQ-musique | squad | 0.9138, 0.9125, 0.9149 | 0.0075 | -0.0003 | -0.0006 | -0.0008 |
| WQ-musique | musique | 0.5190, 0.5227, 0.5381 | 0.0275 | -0.0044 | +0.0054 | +0.0063 |
| WQ-musique | hotpotqa | 0.8460, 0.8469, 0.8491 | 0.0075 | +0.0039 | -0.0054 | -0.0011 |
| WQ-musique | 2wiki | 0.9155, 0.9159, 0.9133 | 0.0075 | +0.0030 | -0.0050 | +0.0033 |
| WQ-musique | webqsp | 0.6457, 0.6498, 0.6552 | 0.0146 | +0.0028 | -0.0074 | -0.0013 |
| WQ-hotpotqa | metaqa | 0.3668, 0.4261, 0.4131 | 0.1064 | +0.0018 | +0.0347 | -0.0018 |
| WQ-hotpotqa | squad | 0.9132, 0.9120, 0.9132 | 0.0075 | +0.0016 | -0.0003 | +0.0025 |
| WQ-hotpotqa | musique | 0.5806, 0.5803, 0.5777 | 0.0075 | +0.0013 | -0.0117* | -0.0066 |
| WQ-hotpotqa | hotpotqa | 0.8193, 0.8182, 0.8136 | 0.0082 | -0.0085* | +0.0009 | -0.0040 |
| WQ-hotpotqa | 2wiki | 0.9135, 0.9166, 0.9135 | 0.0075 | +0.0033 | -0.0053 | -0.0003 |
| WQ-hotpotqa | webqsp | 0.6469, 0.6597, 0.6512 | 0.0191 | -0.0045 | -0.0092 | -0.0051 |
| WQ-hotpotqa | musique-3hop | 0.5351, 0.5333, 0.5263 | 0.0127 | +0.0031 | -0.0167* | -0.0101 |
| WQ-hotpotqa | musique-4hop | 0.3574, 0.3562, 0.3562 | 0.0075 | +0.0043 | -0.0093* | -0.0025 |
