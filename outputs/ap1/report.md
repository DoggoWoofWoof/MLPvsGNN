# AP1 report (docs/AP1_RULE_NAMES_AND_MUSIQUE_REACH.md) - graph level, no model

Share of slots linked by the union of (structural family, K, NER form). Difference = form minus N_sp, paired over questions, 95% interval of 2000 question resamples (seed 0). Calls: ABOVE / LEVEL / BELOW by whether the interval clears 0.

## musique

nodes 11656, questions 1000; names 57404, mentions 108176, df 2-25 9361, df > 25 281; pairs: N_sp_pairs 75002, S_pairs 42823, R_pairs 41429, K_pairs 25722

### musique, world R (union of R + K + the NER form)

| NER form | pairs | NER pairs | new pairs | CH linked | CH share | CH diff vs N_sp [95%] call | GP linked | GP share | GP diff vs N_sp [95%] call | edges / ref |
|---|---:|---:|---:|---:|---:|:--|---:|---:|:--|---:|
| none | 63569 | 0 | 0 | 642/1648 | 38.96% | -29.31 [-31.55, -27.15] BELOW | 658/2462 | 26.73% | -21.32 [-23.03, -19.74] BELOW | 0.506 |
| N_sp | 125662 | 75002 | 62093 | 1125/1648 | 68.26% | +0.00 [+0.00, +0.00] LEVEL | 1183/2462 | 48.05% | +0.00 [+0.00, +0.00] LEVEL | 1.000 |
| CN_u | 131237 | 77961 | 67668 | 1041/1648 | 63.17% | -5.10 [-6.76, -3.43] BELOW | 1103/2462 | 44.80% | -3.25 [-4.51, -2.07] BELOW | 1.044 |
| CN_all | 141965 | 92847 | 78396 | 1146/1648 | 69.54% | +1.27 [-0.36, +2.84] LEVEL | 1210/2462 | 49.15% | +1.10 [-0.08, +2.22] LEVEL | 1.130 |
| CN_wu | 134008 | 81078 | 70439 | 1044/1648 | 63.35% | -4.92 [-6.58, -3.25] BELOW | 1108/2462 | 45.00% | -3.05 [-4.31, -1.89] BELOW | 1.066 |
| CN_w | 147391 | 99554 | 83822 | 1150/1648 | 69.78% | +1.52 [-0.12, +3.10] LEVEL | 1216/2462 | 49.39% | +1.34 [+0.16, +2.45] ABOVE | 1.173 |

Price, R-world, CH: new pairs per slot the NER slot adds over the union without it: N_sp 128.6, CN_u 169.6, CN_all 155.5, CN_wu 175.2, CN_w 165.0
Marginal price, R-world, CH: CN_all_over_none 155.5, CN_w_over_CN_all 1356.5, CN_wu_over_CN_u 923.7, N_sp_over_none 128.6
- CN_u minus CN_all, CH: -6.37 [-7.53, -5.23] BELOW
- CN_wu minus CN_all, CH: -6.19 [-7.37, -5.04] BELOW
- CN_w minus CN_all, CH: +0.24 [+0.06, +0.49] ABOVE
Price, R-world, GP: new pairs per slot the NER slot adds over the union without it: N_sp 118.3, CN_u 152.1, CN_all 142.0, CN_wu 156.5, CN_w 150.2
Marginal price, R-world, GP: CN_all_over_none 142.0, CN_w_over_CN_all 904.3, CN_wu_over_CN_u 554.2, N_sp_over_none 118.3
- CN_u minus CN_all, GP: -4.35 [-5.14, -3.58] BELOW
- CN_wu minus CN_all, GP: -4.14 [-4.95, -3.36] BELOW
- CN_w minus CN_all, GP: +0.24 [+0.08, +0.45] ABOVE

### musique, world S (union of S + K + the NER form)

| NER form | pairs | NER pairs | new pairs | CH linked | CH share | CH diff vs N_sp [95%] call | GP linked | GP share | GP diff vs N_sp [95%] call | edges / ref |
|---|---:|---:|---:|---:|---:|:--|---:|---:|:--|---:|
| none | 65657 | 0 | 0 | 614/1648 | 37.26% | -30.70 [-32.91, -28.52] BELOW | 630/2462 | 25.59% | -22.26 [-23.99, -20.68] BELOW | 0.511 |
| N_sp | 128551 | 75002 | 62894 | 1120/1648 | 67.96% | +0.00 [+0.00, +0.00] LEVEL | 1178/2462 | 47.85% | +0.00 [+0.00, +0.00] LEVEL | 1.000 |
| CN_u | 134153 | 77961 | 68496 | 1033/1648 | 62.68% | -5.28 [-6.96, -3.62] BELOW | 1095/2462 | 44.48% | -3.37 [-4.62, -2.19] BELOW | 1.044 |
| CN_all | 144909 | 92847 | 79252 | 1138/1648 | 69.05% | +1.09 [-0.54, +2.68] LEVEL | 1202/2462 | 48.82% | +0.97 [-0.24, +2.09] LEVEL | 1.127 |
| CN_wu | 136935 | 81078 | 71278 | 1036/1648 | 62.86% | -5.10 [-6.80, -3.44] BELOW | 1100/2462 | 44.68% | -3.17 [-4.43, -2.01] BELOW | 1.065 |
| CN_w | 150343 | 99554 | 84686 | 1142/1648 | 69.30% | +1.33 [-0.31, +2.88] LEVEL | 1208/2462 | 49.07% | +1.22 [+0.04, +2.32] ABOVE | 1.169 |

Price, S-world, CH: new pairs per slot the NER slot adds over the union without it: N_sp 124.3, CN_u 163.5, CN_all 151.2, CN_wu 168.9, CN_w 160.4
Marginal price, S-world, CH: CN_all_over_none 151.2, CN_w_over_CN_all 1358.5, CN_wu_over_CN_u 927.3, N_sp_over_none 124.3
- CN_u minus CN_all, CH: -6.37 [-7.53, -5.23] BELOW
- CN_wu minus CN_all, CH: -6.19 [-7.37, -5.04] BELOW
- CN_w minus CN_all, CH: +0.24 [+0.06, +0.49] ABOVE
Price, S-world, GP: new pairs per slot the NER slot adds over the union without it: N_sp 114.8, CN_u 147.3, CN_all 138.6, CN_wu 151.7, CN_w 146.5
Marginal price, S-world, GP: CN_all_over_none 138.6, CN_w_over_CN_all 905.7, CN_wu_over_CN_u 556.4, N_sp_over_none 114.8
- CN_u minus CN_all, GP: -4.35 [-5.14, -3.58] BELOW
- CN_wu minus CN_all, GP: -4.14 [-4.95, -3.36] BELOW
- CN_w minus CN_all, GP: +0.24 [+0.08, +0.45] ABOVE

### musique, by question kind (R-world; descriptive, amendment A1): linked/slots, share, call against N_sp

| NER form | 2hop | 3hop1 | 3hop2 | 4hop1 | 4hop2 | 4hop3 |
|---|---:|---:|---:|---:|---:|---:|
| none | 269/518 51.9% BELOW | 148/729 20.3% BELOW | 37/219 16.9% BELOW | 152/648 23.5% BELOW | 19/162 11.7% BELOW | 33/186 17.7% BELOW |
| N_sp | 414/518 79.9% LEVEL | 297/729 40.7% LEVEL | 96/219 43.8% LEVEL | 257/648 39.7% LEVEL | 63/162 38.9% LEVEL | 56/186 30.1% LEVEL |
| CN_u | 407/518 78.6% LEVEL | 259/729 35.5% BELOW | 71/219 32.4% BELOW | 254/648 39.2% LEVEL | 49/162 30.2% BELOW | 63/186 33.9% ABOVE |
| CN_all | 426/518 82.2% LEVEL | 292/729 40.0% LEVEL | 86/219 39.3% LEVEL | 279/648 43.1% ABOVE | 62/162 38.3% LEVEL | 65/186 35.0% ABOVE |
| CN_wu | 407/518 78.6% LEVEL | 261/729 35.8% BELOW | 74/219 33.8% BELOW | 254/648 39.2% LEVEL | 49/162 30.2% BELOW | 63/186 33.9% ABOVE |
| CN_w | 426/518 82.2% LEVEL | 295/729 40.5% LEVEL | 89/219 40.6% LEVEL | 279/648 43.1% ABOVE | 62/162 38.3% LEVEL | 65/186 35.0% ABOVE |

### musique, reach against NER pairs kept (R-world; the form's pairs by weight, top fraction f; descriptive, amendment A1)

| form | f | NER pairs | min weight | union pairs | CH | GP |
|---|---:|---:|---:|---:|---:|---:|
| N_sp | 0.1 | 7501 | 0.33333 | 68186 | 43.81% | 30.10% |
| N_sp | 0.2 | 15001 | 0.2 | 73876 | 48.85% | 33.47% |
| N_sp | 0.3 | 22501 | 0.14286 | 79948 | 51.64% | 35.46% |
| N_sp | 0.5 | 37501 | 0.08333 | 92659 | 57.28% | 39.60% |
| N_sp | 0.7 | 52502 | 0.05882 | 105562 | 60.92% | 42.65% |
| N_sp | 1.0 | 75002 | 0.04 | 125662 | 68.26% | 48.05% |
| CN_all | 0.1 | 9285 | 0.33333 | 69364 | 46.78% | 32.09% |
| CN_all | 0.2 | 18570 | 0.2 | 76608 | 50.49% | 34.61% |
| CN_all | 0.3 | 27855 | 0.125 | 84170 | 53.70% | 36.80% |
| CN_all | 0.5 | 46424 | 0.07692 | 100246 | 58.80% | 40.74% |
| CN_all | 0.7 | 64993 | 0.05556 | 116605 | 62.99% | 43.87% |
| CN_all | 1.0 | 92847 | 0.04 | 141965 | 69.54% | 49.15% |

### musique, Part B: the bridge name (labels read as a diagnostic only)

1648 chain slots; the rule finds the bridge name in both passages in 1133, in one in 398, in neither in 117. Linked by R+N_sp+K: 1125; by R+CN_all+K: 1146; by R+CN_w+K: 1150. Ceiling of any shared-name rule (any df): 1133; at df <= 100: 1106.
Slots R+N_sp+K misses: 523; of those the bridge name is in both passages in 304 (by df band {'df 101+': 26, 'df 26-50': 112, 'df 2-25': 82, 'df 51-100': 84}), and not in both in 219.

### musique, Part C, GP: slots linked by R+N_sp+K and not by R+CN_w+K (loss) 78, the reverse (gain) 111; loss slots whose passages share a rule name at any df: 42
- Governor of Vatican City | St. Peter's Basilica | shared: ['italian']
- Jessie Woodrow Wilson Sayre | Treaty of Versailles | shared: []
- Bustami | Indonesia–Timor Leste Commission of Truth and Friendship | shared: ['july']
- Guangling District | Pfrang Association | shared: ['china']
- Middle Ages | Holy Roman Emperor | shared: []
- Seria, Belait | The Biggest Loser Brunei: Lose It All | shared: ['march']
- Darren Carter | Second City derby | shared: []
- Jessie Woodrow Wilson Sayre | Prohibition in the United States | shared: []
- Qiantong, Zhejiang | Hu Sanxing | shared: []
- Indonesia–Timor Leste Commission of Truth and Friendship | Kiwil | shared: ['indonesia']
- Prague underground (culture) | Josip Broz Tito | shared: []
- Warner Records | Back of My Mind | shared: ['warner bros']

### musique, Part C, GPk:2hop: slots linked by R+N_sp+K and not by R+CN_w+K (loss) 14, the reverse (gain) 26; loss slots whose passages share a rule name at any df: 2
- Jessie Woodrow Wilson Sayre | Prohibition in the United States | shared: []
- Qiantong, Zhejiang | Hu Sanxing | shared: []
- Reresby Sitwell | The Sitwells | shared: []
- The Mickey Mouse Club | Mickey's Safari in Letterland | shared: []
- Jessie Woodrow Wilson Sayre | State of the Union | shared: []
- Middle Ages | Arthur's Seat | shared: []
- Plymouth Notch Cemetery | 1924 United States presidential election | shared: []
- M. King Hubbert | Royal Dutch Shell | shared: ['october']
- The Motorcycle Diaries (film) | San Clemente, Chile | shared: []
- Walker, Texas Ranger | Forest Warrior | shared: []
- Texian Army | Sam Houston National Forest | shared: ['texas']
- The Code of Conduct and the Vietnam Prisoners of War | 2008 United States presidential election | shared: []

### musique, Part C, GPk:3hop1: slots linked by R+N_sp+K and not by R+CN_w+K (loss) 38, the reverse (gain) 36; loss slots whose passages share a rule name at any df: 22
- Governor of Vatican City | St. Peter's Basilica | shared: ['italian']
- Jessie Woodrow Wilson Sayre | Treaty of Versailles | shared: []
- Bustami | Indonesia–Timor Leste Commission of Truth and Friendship | shared: ['july']
- Seria, Belait | The Biggest Loser Brunei: Lose It All | shared: ['march']
- Indonesia–Timor Leste Commission of Truth and Friendship | Kiwil | shared: ['indonesia']
- Warner Records | Back of My Mind | shared: ['warner bros']
- Warner Records | Ain't Living Long Like This | shared: ['warner bros']
- 1973 oil crisis | Daihatsu Boon | shared: ['japanese']
- 1973 oil crisis | Lexus RX | shared: []
- Warner Records | Atomic Playboys | shared: ['u.s', 'warner bros']
- Bang Bon District | Bodindecha | shared: []
- Governor of Vatican City | St. Peter's Basilica | shared: ['italian']

### musique, Part C, GPk:3hop2: slots linked by R+N_sp+K and not by R+CN_w+K (loss) 22, the reverse (gain) 15; loss slots whose passages share a rule name at any df: 15
- Guangling District | Pfrang Association | shared: ['china']
- Middle Ages | Holy Roman Emperor | shared: []
- Darren Carter | Second City derby | shared: []
- Second City derby | Tony Richards (footballer) | shared: ['march', 'may']
- Second City derby | Matthew Webb (footballer) | shared: ['march', 'september']
- 1973 oil crisis | 1967 Italian Grand Prix | shared: []
- John Phan | South Central Coast | shared: ['vietnam']
- Second City derby | Gary Rowett | shared: ['december', 'march', 'may', 'october']
- John Phan | South Central Coast | shared: ['vietnam']
- John Phan | South Central Coast | shared: ['vietnam']
- John Phan | South Central Coast | shared: ['vietnam']
- Second City derby | Krystian Pearce | shared: []

### musique, Part C, GPk:4hop1: slots linked by R+N_sp+K and not by R+CN_w+K (loss) 3, the reverse (gain) 25; loss slots whose passages share a rule name at any df: 3
- Major League Baseball Most Valuable Player Award | Opening Day | shared: ['world series']
- Major League Baseball Most Valuable Player Award | 2017 Major League Baseball All-Star Game | shared: ['world series']
- List of Super Bowl halftime shows | Publishers Clearing House | shared: ['super bowl']

### musique, Part C, GPk:4hop2: slots linked by R+N_sp+K and not by R+CN_w+K (loss) 1, the reverse (gain) 0; loss slots whose passages share a rule name at any df: 0
- Prague underground (culture) | Josip Broz Tito | shared: []

### musique, Part C, GPk:4hop3: slots linked by R+N_sp+K and not by R+CN_w+K (loss) 0, the reverse (gain) 9; loss slots whose passages share a rule name at any df: 0

### musique, Part C, CH: slots linked by R+N_sp+K and not by R+CN_w+K (loss) 71, the reverse (gain) 96; loss slots whose passages share a rule name at any df: 38
- Governor of Vatican City | St. Peter's Basilica | shared: ['italian']
- Jessie Woodrow Wilson Sayre | Treaty of Versailles | shared: []
- Bustami | Indonesia–Timor Leste Commission of Truth and Friendship | shared: ['july']
- Middle Ages | Holy Roman Emperor | shared: []
- Seria, Belait | The Biggest Loser Brunei: Lose It All | shared: ['march']
- Darren Carter | Second City derby | shared: []
- Jessie Woodrow Wilson Sayre | Prohibition in the United States | shared: []
- Qiantong, Zhejiang | Hu Sanxing | shared: []
- Indonesia–Timor Leste Commission of Truth and Friendship | Kiwil | shared: ['indonesia']
- Warner Records | Back of My Mind | shared: ['warner bros']
- Second City derby | Tony Richards (footballer) | shared: ['march', 'may']
- Warner Records | Ain't Living Long Like This | shared: ['warner bros']

## 2wiki

nodes 6119, questions 1000; names 35074, mentions 61941, df 2-25 6439, df > 25 107; pairs: N_sp_pairs 36147, S_pairs 1984, R_pairs 2030, K_pairs 13415

### 2wiki, world R (union of R + K + the NER form)

| NER form | pairs | NER pairs | new pairs | GP linked | GP share | GP diff vs N_sp [95%] call | edges / ref |
|---|---:|---:|---:|---:|---:|:--|---:|
| none | 14206 | 0 | 0 | 956/2175 | 43.95% | -2.30 [-2.98, -1.69] BELOW | 0.309 |
| N_sp | 45980 | 36147 | 31774 | 1006/2175 | 46.25% | +0.00 [+0.00, +0.00] LEVEL | 1.000 |
| CN_u | 49689 | 39509 | 35483 | 976/2175 | 44.87% | -1.38 [-1.93, -0.86] BELOW | 1.081 |
| CN_all | 50897 | 41593 | 36691 | 1006/2175 | 46.25% | +0.00 [-0.18, +0.19] LEVEL | 1.107 |
| CN_wu | 51287 | 41180 | 37081 | 976/2175 | 44.87% | -1.38 [-1.93, -0.86] BELOW | 1.115 |
| CN_w | 52490 | 43254 | 38284 | 1006/2175 | 46.25% | +0.00 [-0.18, +0.19] LEVEL | 1.142 |

Price, R-world, GP: new pairs per slot the NER slot adds over the union without it: N_sp 635.5, CN_u 1774.2, CN_all 733.8, CN_wu 1854.0, CN_w 765.7
Marginal price, R-world, GP: CN_all_over_none 733.8, CN_w_over_CN_all None, CN_wu_over_CN_u None, N_sp_over_none 635.5
- CN_u minus CN_all, GP: -1.38 [-1.92, -0.88] BELOW
- CN_wu minus CN_all, GP: -1.38 [-1.92, -0.88] BELOW
- CN_w minus CN_all, GP: +0.00 [+0.00, +0.00] LEVEL

### 2wiki, world S (union of S + K + the NER form)

| NER form | pairs | NER pairs | new pairs | GP linked | GP share | GP diff vs N_sp [95%] call | edges / ref |
|---|---:|---:|---:|---:|---:|:--|---:|
| none | 14104 | 0 | 0 | 977/2175 | 44.92% | -1.29 [-1.81, -0.80] BELOW | 0.307 |
| N_sp | 45861 | 36147 | 31757 | 1005/2175 | 46.21% | +0.00 [+0.00, +0.00] LEVEL | 1.000 |
| CN_u | 49581 | 39509 | 35477 | 996/2175 | 45.79% | -0.41 [-0.77, -0.09] BELOW | 1.081 |
| CN_all | 50817 | 41593 | 36713 | 1005/2175 | 46.21% | +0.00 [-0.18, +0.19] LEVEL | 1.108 |
| CN_wu | 51180 | 41180 | 37076 | 996/2175 | 45.79% | -0.41 [-0.77, -0.09] BELOW | 1.116 |
| CN_w | 52410 | 43254 | 38306 | 1005/2175 | 46.21% | +0.00 [-0.18, +0.19] LEVEL | 1.143 |

Price, S-world, GP: new pairs per slot the NER slot adds over the union without it: N_sp 1134.2, CN_u 1867.2, CN_all 1311.2, CN_wu 1951.4, CN_w 1368.1
Marginal price, S-world, GP: CN_all_over_none 1311.2, CN_w_over_CN_all None, CN_wu_over_CN_u None, N_sp_over_none 1134.2
- CN_u minus CN_all, GP: -0.41 [-0.70, -0.18] BELOW
- CN_wu minus CN_all, GP: -0.41 [-0.70, -0.18] BELOW
- CN_w minus CN_all, GP: +0.00 [+0.00, +0.00] LEVEL

### 2wiki, by question kind (R-world; descriptive, amendment A1): linked/slots, share, call against N_sp

| NER form | bridge_comparison | comparison | compositional | inference |
|---|---:|---:|---:|---:|
| none | 443/1410 31.4% BELOW | 12/244 4.9% BELOW | 397/413 96.1% BELOW | 104/108 96.3% BELOW |
| N_sp | 474/1410 33.6% LEVEL | 16/244 6.6% LEVEL | 408/413 98.8% LEVEL | 108/108 100.0% LEVEL |
| CN_u | 451/1410 32.0% BELOW | 16/244 6.6% LEVEL | 401/413 97.1% BELOW | 108/108 100.0% LEVEL |
| CN_all | 474/1410 33.6% LEVEL | 16/244 6.6% LEVEL | 408/413 98.8% LEVEL | 108/108 100.0% LEVEL |
| CN_wu | 451/1410 32.0% BELOW | 16/244 6.6% LEVEL | 401/413 97.1% BELOW | 108/108 100.0% LEVEL |
| CN_w | 474/1410 33.6% LEVEL | 16/244 6.6% LEVEL | 408/413 98.8% LEVEL | 108/108 100.0% LEVEL |

### 2wiki, reach against NER pairs kept (R-world; the form's pairs by weight, top fraction f; descriptive, amendment A1)

| form | f | NER pairs | min weight | union pairs | GP |
|---|---:|---:|---:|---:|---:|
| N_sp | 0.1 | 3615 | 0.33333 | 16021 | 45.20% |
| N_sp | 0.2 | 7230 | 0.25 | 18851 | 45.66% |
| N_sp | 0.3 | 10845 | 0.16667 | 21922 | 45.70% |
| N_sp | 0.5 | 18074 | 0.09091 | 28517 | 46.07% |
| N_sp | 0.7 | 25303 | 0.05556 | 35475 | 46.21% |
| N_sp | 1.0 | 36147 | 0.04 | 45980 | 46.25% |
| CN_all | 0.1 | 4160 | 0.39286 | 16295 | 45.10% |
| CN_all | 0.2 | 8319 | 0.25 | 19505 | 45.52% |
| CN_all | 0.3 | 12478 | 0.16667 | 23074 | 45.66% |
| CN_all | 0.5 | 20797 | 0.1 | 30730 | 46.11% |
| CN_all | 0.7 | 29116 | 0.0625 | 38783 | 46.16% |
| CN_all | 1.0 | 41593 | 0.04 | 50897 | 46.25% |

### 2wiki, Part C, GP: slots linked by R+N_sp+K and not by R+CN_w+K (loss) 2, the reverse (gain) 2; loss slots whose passages share a rule name at any df: 2
- St. Mary's School, Patiala | Apeejay Education Society | shared: ['india']
- Michael Curtiz | Ralph Nelson | shared: ['academy awards', 'december']

### 2wiki, Part C, GPk:bridge_comparison: slots linked by R+N_sp+K and not by R+CN_w+K (loss) 1, the reverse (gain) 1; loss slots whose passages share a rule name at any df: 1
- Michael Curtiz | Ralph Nelson | shared: ['academy awards', 'december']

### 2wiki, Part C, GPk:comparison: slots linked by R+N_sp+K and not by R+CN_w+K (loss) 1, the reverse (gain) 1; loss slots whose passages share a rule name at any df: 1
- St. Mary's School, Patiala | Apeejay Education Society | shared: ['india']

### 2wiki, Part C, GPk:compositional: slots linked by R+N_sp+K and not by R+CN_w+K (loss) 0, the reverse (gain) 0; loss slots whose passages share a rule name at any df: 0

### 2wiki, Part C, GPk:inference: slots linked by R+N_sp+K and not by R+CN_w+K (loss) 0, the reverse (gain) 0; loss slots whose passages share a rule name at any df: 0

## hotpotqa

nodes 9811, questions 1000; names 57894, mentions 106524, df 2-25 9997, df > 25 206; pairs: N_sp_pairs 68698, S_pairs 4400, R_pairs 4315, K_pairs 21805

### hotpotqa, world R (union of R + K + the NER form)

| NER form | pairs | NER pairs | new pairs | GP linked | GP share | GP diff vs N_sp [95%] call | edges / ref |
|---|---:|---:|---:|---:|---:|:--|---:|
| none | 23582 | 0 | 0 | 730/1000 | 73.00% | -4.50 [-5.80, -3.20] BELOW | 0.283 |
| N_sp | 83356 | 68698 | 59774 | 775/1000 | 77.50% | +0.00 [+0.00, +0.00] LEVEL | 1.000 |
| CN_u | 91958 | 76278 | 68376 | 774/1000 | 77.40% | -0.10 [-1.10, +0.90] LEVEL | 1.103 |
| CN_all | 98675 | 85369 | 75093 | 791/1000 | 79.10% | +1.60 [+0.70, +2.60] ABOVE | 1.184 |
| CN_wu | 96539 | 81062 | 72957 | 775/1000 | 77.50% | +0.00 [-1.00, +1.00] LEVEL | 1.158 |
| CN_w | 103537 | 90411 | 79955 | 792/1000 | 79.20% | +1.70 [+0.80, +2.70] ABOVE | 1.242 |

Price, R-world, GP: new pairs per slot the NER slot adds over the union without it: N_sp 1328.3, CN_u 1554.0, CN_all 1231.0, CN_wu 1621.3, CN_w 1289.6
Marginal price, R-world, GP: CN_all_over_none 1231.0, CN_w_over_CN_all 4862.0, CN_wu_over_CN_u 4581.0, N_sp_over_none 1328.3
- CN_u minus CN_all, GP: -1.70 [-2.60, -1.00] BELOW
- CN_wu minus CN_all, GP: -1.60 [-2.50, -0.80] BELOW
- CN_w minus CN_all, GP: +0.10 [+0.00, +0.30] LEVEL

### hotpotqa, world S (union of S + K + the NER form)

| NER form | pairs | NER pairs | new pairs | GP linked | GP share | GP diff vs N_sp [95%] call | edges / ref |
|---|---:|---:|---:|---:|---:|:--|---:|
| none | 23400 | 0 | 0 | 827/1000 | 82.70% | -0.60 [-1.10, -0.20] BELOW | 0.281 |
| N_sp | 83188 | 68698 | 59788 | 833/1000 | 83.30% | +0.00 [+0.00, +0.00] LEVEL | 1.000 |
| CN_u | 91745 | 76278 | 68345 | 837/1000 | 83.70% | +0.40 [+0.00, +0.90] LEVEL | 1.103 |
| CN_all | 98540 | 85369 | 75140 | 837/1000 | 83.70% | +0.40 [+0.00, +0.90] LEVEL | 1.185 |
| CN_wu | 96323 | 81062 | 72923 | 838/1000 | 83.80% | +0.50 [+0.00, +1.00] LEVEL | 1.158 |
| CN_w | 103401 | 90411 | 80001 | 838/1000 | 83.80% | +0.50 [+0.00, +1.00] LEVEL | 1.243 |

Price, S-world, GP: new pairs per slot the NER slot adds over the union without it: N_sp 9964.7, CN_u 6834.5, CN_all 7514.0, CN_wu 6629.4, CN_w 7272.8
Marginal price, S-world, GP: CN_all_over_none 7514.0, CN_w_over_CN_all 4861.0, CN_wu_over_CN_u 4578.0, N_sp_over_none 9964.7
- CN_u minus CN_all, GP: +0.00 [+0.00, +0.00] LEVEL
- CN_wu minus CN_all, GP: +0.10 [+0.00, +0.30] LEVEL
- CN_w minus CN_all, GP: +0.10 [+0.00, +0.30] LEVEL

### hotpotqa, by question kind (R-world; descriptive, amendment A1): linked/slots, share, call against N_sp

| NER form | bridge | comparison |
|---|---:|---:|
| none | 714/811 88.0% BELOW | 16/189 8.5% BELOW |
| N_sp | 753/811 92.8% LEVEL | 22/189 11.6% LEVEL |
| CN_u | 748/811 92.2% LEVEL | 26/189 13.8% LEVEL |
| CN_all | 765/811 94.3% ABOVE | 26/189 13.8% LEVEL |
| CN_wu | 748/811 92.2% LEVEL | 27/189 14.3% LEVEL |
| CN_w | 765/811 94.3% ABOVE | 27/189 14.3% LEVEL |

### hotpotqa, reach against NER pairs kept (R-world; the form's pairs by weight, top fraction f; descriptive, amendment A1)

| form | f | NER pairs | min weight | union pairs | GP |
|---|---:|---:|---:|---:|---:|
| N_sp | 0.1 | 6870 | 0.33333 | 27332 | 74.40% |
| N_sp | 0.2 | 13740 | 0.21667 | 32624 | 75.20% |
| N_sp | 0.3 | 20610 | 0.15 | 38411 | 75.50% |
| N_sp | 0.5 | 34349 | 0.1 | 50236 | 76.50% |
| N_sp | 0.7 | 48089 | 0.0625 | 63234 | 77.00% |
| N_sp | 1.0 | 68698 | 0.04 | 83356 | 77.50% |
| CN_all | 0.1 | 8537 | 0.33333 | 28254 | 75.30% |
| CN_all | 0.2 | 17074 | 0.20202 | 35004 | 75.90% |
| CN_all | 0.3 | 25611 | 0.14286 | 42487 | 76.90% |
| CN_all | 0.5 | 42685 | 0.09091 | 57453 | 77.60% |
| CN_all | 0.7 | 59759 | 0.0625 | 73711 | 78.40% |
| CN_all | 1.0 | 85369 | 0.04 | 98675 | 79.10% |

### hotpotqa, Part C, GP: slots linked by R+N_sp+K and not by R+CN_w+K (loss) 3, the reverse (gain) 20; loss slots whose passages share a rule name at any df: 1
- Joseph McCarthy | James Gregory (actor) | shared: ['american']
- Gender | Mary Karooro Okurut | shared: []
- Wire (band) | Young the Giant | shared: []

### hotpotqa, Part C, GPk:bridge: slots linked by R+N_sp+K and not by R+CN_w+K (loss) 2, the reverse (gain) 14; loss slots whose passages share a rule name at any df: 1
- Joseph McCarthy | James Gregory (actor) | shared: ['american']
- Gender | Mary Karooro Okurut | shared: []

### hotpotqa, Part C, GPk:comparison: slots linked by R+N_sp+K and not by R+CN_w+K (loss) 1, the reverse (gain) 6; loss slots whose passages share a rule name at any df: 0
- Wire (band) | Young the Giant | shared: []

