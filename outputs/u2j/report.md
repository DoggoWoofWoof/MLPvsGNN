# U2j report (docs/U2J_LINK_PRECISION.md)

Outcome: **ADOPT_FOR_POOLS**, best R4

Score (mean I's ALL, musique, hotpotqa, 2wiki): R0 0.8368, R1 0.839, R2 0.8435, R3 0.8438, R4 0.851

## musique, s1sel (1534 questions)

117,534 rows, 4,687,840 entries; classes {'mutual': 166846, 'out': 2137570, 'in': 2137570, 'sibling': 245854}; lead 2,529,516; W2_S mean degree 39.88.

| order | I's ALL | recall | pool | bridge@5 | bridge@25 | gap share (I, bridge@25) |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | 0.7751 | 0.9122 | 2067.2 | 0.1558 | 0.1991 | None, None |
| R1 | 0.7777 | 0.9127 | 2074.3 | 0.1569 | 0.2067 | None, None |
| R2 | 0.779 | 0.9133 | 2073.1 | 0.1623 | 0.21 | None, None |
| R3 | 0.7686 | 0.9103 | 2057.6 | 0.184 | 0.2284 | None, None |
| R4 | 0.7849 | 0.9149 | 2071.0 | 0.1916 | 0.2219 | None, None |
| PKG | 0.7679 | 0.9098 | 2093.5 | 0.145 | 0.1926 | None, None |

Beside (musique): 2,304,416 W2 links, 0.9991 hyperlinked. AUC: lead 0.5035, pos 0.5442, count 0.5482, rare 0.1467, caps 0.8011, capmid 0.5565, tokens 0.2091, alias None

## squad, s1sel (1498 questions)

8,329 rows, 710,478 entries; classes {'mutual': 93933, 'out': 194380, 'in': 176119, 'sibling': 246046}; lead 232,335; W2_S mean degree 86.81.

| order | I's ALL | recall | pool | bridge@5 | bridge@25 | gap share (I, bridge@25) |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | None | None | None | 0.0 | 0.0 | None, None |
| R1 | None | None | None | 0.0 | 0.0 | None, None |
| R2 | None | None | None | 0.0 | 0.0 | None, None |
| R3 | None | None | None | 0.0 | 0.0 | None, None |
| R4 | None | None | None | 0.0 | 0.0 | None, None |
| PKG | None | None | None | 0.0 | 0.0 | None, None |

Beside (squad): 685,845 W2 links, 0.726 hyperlinked. AUC: lead 0.4931, pos 0.5036, count 0.6008, rare 0.2269, caps 0.7311, capmid 0.527, tokens 0.0016, alias None

## hotpotqa, s1sel (1508 questions)

12,177 rows, 276,156 entries; classes {'mutual': 2381, 'out': 115831, 'in': 157944, 'sibling': 0}; lead 145,852; W2_S mean degree 12.29.

| order | I's ALL | recall | pool | bridge@5 | bridge@25 | gap share (I, bridge@25) |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | 0.8985 | 0.944 | 119.6 | 0.3283 | 0.6698 | None, None |
| R1 | 0.8985 | 0.944 | 118.7 | 0.3397 | 0.6717 | 0.0, 0.008 |
| R2 | 0.8992 | 0.9443 | 118.5 | 0.3226 | 0.6736 | 0.01, 0.016 |
| R3 | 0.8985 | 0.9443 | 120.4 | 0.5598 | 0.6698 | 0.0, 0.0 |
| R4 | 0.8999 | 0.9446 | 118.6 | 0.463 | 0.6736 | 0.02, 0.016 |
| PKG | 0.9682 | 0.9791 | 94.1 | 0.7116 | 0.9013 | None, None |

Beside (hotpotqa): 32,334,355 W2 links, 0.3236 hyperlinked. AUC: lead 0.4828, pos 0.5863, count 0.4861, rare 0.7412, caps 0.5377, capmid 0.5682, tokens 0.7197, alias None

## 2wiki, s1sel (1496 questions)

11,973 rows, 282,445 entries; classes {'mutual': 3388, 'out': 214874, 'in': 64181, 'sibling': 2}; lead 117,130; W2_S mean degree 23.87.

| order | I's ALL | recall | pool | bridge@5 | bridge@25 | gap share (I, bridge@25) |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | 0.8369 | 0.9345 | 162.2 | 0.3181 | 0.8133 | None, None |
| R1 | 0.8409 | 0.9362 | 161.7 | 0.3357 | 0.8213 | 0.067, 0.105 |
| R2 | 0.8523 | 0.9405 | 160.8 | 0.4688 | 0.8397 | 0.259, 0.347 |
| R3 | 0.8643 | 0.9462 | 165.1 | 0.6418 | 0.8486 | 0.461, 0.464 |
| R4 | 0.8683 | 0.9479 | 162.3 | 0.7051 | 0.8598 | 0.528, 0.611 |
| PKG | 0.8964 | 0.9579 | 105.1 | 0.6987 | 0.8894 | None, None |

Beside (2wiki): 71,913,824 W2 links, 0.2305 hyperlinked. AUC: lead 0.4869, pos 0.5499, count 0.4825, rare 0.7715, caps 0.5121, capmid 0.5474, tokens 0.7479, alias None
