# D3 data scale: zrc on a stride of its fit carves

**DATA_LIMITED** (DATA_LIMITED if the half fits read LOSS (seed null) on >= 2 reads; docs/DIAG_DATA_SCALE.md).

| split | dataset | read | zrc R@5 | half delta | half call | quarter delta | quarter call |
| --- | --- | --- | ---: | ---: | --- | ---: | --- |
| L-musique | metaqa | in-domain | 0.7782 | -0.0043 | WITHIN | -0.0259 | LOSS |
| L-musique | squad | in-domain | 0.9105 | -0.0019 | WITHIN | -0.0084 | LOSS |
| L-musique | musique | zero-shot | 0.5235 | -0.0071 | WITHIN | -0.0180 | WITHIN |
| L-musique | hotpotqa | in-domain | 0.9041 | -0.0068 | WITHIN | -0.0146 | LOSS |
| L-musique | 2wiki | in-domain | 0.8733 | -0.0078 | LOSS | -0.0175 | LOSS |
| L-musique | webqsp | zero-shot | 0.3384 | -0.0310 | LOSS | -0.0813 | LOSS |
| L-hotpotqa | metaqa | in-domain | 0.7793 | -0.0064 | WITHIN | -0.0298 | LOSS |
| L-hotpotqa | squad | in-domain | 0.9112 | -0.0015 | WITHIN | -0.0089 | LOSS |
| L-hotpotqa | musique | in-domain | 0.5645 | +0.0036 | WITHIN | -0.0093 | LOSS |
| L-hotpotqa | hotpotqa | zero-shot | 0.8451 | -0.0018 | WITHIN | -0.0084 | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.8731 | -0.0118 | LOSS | -0.0242 | LOSS |
| L-hotpotqa | webqsp | zero-shot | 0.2865 | -0.0381 | LOSS | -0.0542 | LOSS |

Half: 4 LOSS, 0 GAIN of 12 reads, mean delta -0.0096.
Quarter: 10 LOSS, 0 GAIN of 12 reads, mean delta -0.0250.
