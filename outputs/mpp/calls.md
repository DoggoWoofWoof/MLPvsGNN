# MPP: zrckpp (zrckp + LTW's next-hop columns and GNP's six pair columns (NEXT: LTM's seven, then pair_max, res_max, pair_mean, res_mean, pair_1 and res_1 of each row against rrf's top five)) against LTM's zrckpn, on PKO's pools, on the host's CPU

## Screen: NO_GAIN

counts: {"GAIN": 0, "LOSS": 0, "WITHIN": 10}

| split | read | zrckpn s0, s1, s2 | zrckpp s0 | delta | floor | call |
|---|---|---|---:|---:|---:|---|
| WQ-musique | squad | 0.9138, 0.9125, 0.9149 | 0.9139 | +0.0001 | 0.0075 | WITHIN |
| WQ-musique | hotpotqa | 0.8460, 0.8469, 0.8491 | 0.8452 | -0.0008 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | 0.9155, 0.9159, 0.9133 | 0.9150 | -0.0005 | 0.0075 | WITHIN |
| WQ-musique | webqsp | 0.6457, 0.6498, 0.6552 | 0.6473 | +0.0016 | 0.0146 | WITHIN |
| WQ-hotpotqa | squad | 0.9132, 0.9120, 0.9132 | 0.9141 | +0.0009 | 0.0075 | WITHIN |
| WQ-hotpotqa | musique | 0.5806, 0.5803, 0.5777 | 0.5794 | -0.0012 | 0.0075 | WITHIN |
| WQ-hotpotqa | 2wiki | 0.9135, 0.9166, 0.9135 | 0.9146 | +0.0011 | 0.0075 | WITHIN |
| WQ-hotpotqa | webqsp | 0.6469, 0.6597, 0.6512 | 0.6534 | +0.0065 | 0.0191 | WITHIN |
| WQ-hotpotqa | musique-3hop | 0.5351, 0.5333, 0.5263 | 0.5316 | -0.0035 | 0.0127 | WITHIN |
| WQ-hotpotqa | musique-4hop | 0.3574, 0.3562, 0.3562 | 0.3556 | -0.0019 | 0.0075 | WITHIN |
| WQ-musique | musique | 0.5190, 0.5227, 0.5381 | 0.5510 | +0.0320 | 0.0275 | GAIN (beside) |
| WQ-musique | metaqa | 0.4126, 0.4256, 0.4167 | 0.4014 | -0.0112 | 0.0193 | WITHIN (beside) |
| WQ-hotpotqa | hotpotqa | 0.8193, 0.8182, 0.8136 | 0.8226 | +0.0033 | 0.0082 | WITHIN (beside) |
| WQ-hotpotqa | metaqa | 0.3668, 0.4261, 0.4131 | 0.3848 | +0.0180 | 0.1064 | WITHIN (beside) |
| WQ-musique | musique-3hop | 0.4544, 0.4596, 0.4702 | 0.4904 | +0.0360 | 0.0235 | GAIN (beside) |
| WQ-musique | musique-4hop | 0.3086, 0.3148, 0.3352 | 0.3259 | +0.0173 | 0.0385 | WITHIN (beside) |

next (section 7): the six pair columns beside NEXT close for the MLP; next form: the same six scores against rrf's top ten rows

Beside, deciding nothing, as filed: GNP screen INCOMPLETE, full run None; LTA screen NOT_ADOPTED, full run None; LTS screen NO_GAIN, full run None

| fit | train s | read s |
|---|---:|---:|
| scr-mpp-zrckpp-s0 | 2676.0 | 4263.8 |
| scr-mpp-zrckpp-s0-hp | 4682.1 | 4257.9 |
