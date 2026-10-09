# U1b: one link rule on all six full corpora

## Hyperlink recovery

| dataset | rule edges | hyperlinks | directed P | directed R | unordered P | unordered R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wiki | 205,604,282 | 28,963,600 | 0.0993 | 0.7048 | 0.0978 | 0.7111 |
| hotpotqa | 76,327,595 | 15,367,541 | 0.1384 | 0.6876 | 0.1371 | 0.6915 |

## Graphs

| dataset | nodes | package structural (edges / isolated) | mentions (edges / isolated / max in) | structural_U (edges / isolated) | check |
| --- | ---: | --- | --- | --- | --- |
| musique | 117,534 | 2,744,076 / 0.0839 | 2,744,076 / 0.0839 / 14,152 | 2,744,076 / 0.0839 | identity EQUAL |
| squad | 20,233 | 874,190 / 0.3138 | 874,190 / 0.3138 / 2,433 | 874,190 / 0.3138 | identity EQUAL |
| 2wiki | 5,989,847 | 28,963,600 / 0.0861 | 205,604,282 / 0.0017 / 1,582,529 | 205,604,282 / 0.0017 | hyperlinks dropped |
| hotpotqa | 5,233,329 | 15,367,541 / 0.1054 | 76,327,595 / 0.0006 / 552,106 | 76,327,595 / 0.0006 | hyperlinks dropped |
| metaqa | 43,234 | 133,582 / 0.0 | 26,938 / 0.5252 / 577 | 160,520 / 0.0 | mention pairs already KB pairs 0.0417 |
| webqsp | 2,592,894 | 8,309,195 / 0.0 | 321,938,109 / 0.042 / 1,588,545 | 330,247,304 / 0.0 | mention pairs already KB pairs 0.0141 |

## Most-mentioned titles

- **musique**: United States (14,152); United States (14,151); United States (14,151); United States (14,151); United States (14,151); United States (14,151); United States (14,151); Time (8,050); Time (8,050); Time (8,049)
- **squad**: Time (2,433); Time (2,433); Time (2,432); Time (2,432); Time (2,432); Time (2,432); Time (2,432); Time (2,432); Time (2,432); Time (2,432)
- **2wiki**: From (1,582,529); With (1,533,469); Also (1,050,367); That (974,960); Born (937,047); Which (890,858); First (784,185); It Was (686,043); This (637,512); American (621,588)
- **hotpotqa**: That (552,106); Which (474,717); It Was (469,148); First (421,337); County (376,467); District (359,616); Family (321,829); South (307,882); This (298,820); United States (292,059)
- **metaqa**: john (577); Michael (437); Paul (234); love (219); Love (219); night (151); Jack (145); Frank (136); life (125); Life (125)
- **webqsp**: Record (1,588,545); Topic (447,858); Film (380,395); film (380,395); Region (300,673); Statistical region (275,295); Performance (207,089); Notable for (195,745); Common (121,769); Webpage (121,318)
