# zkind's fits against zrc's (the next base): **NOT_ADOPTED**

R@5 of zkind's p@swa minus zrc's p@swa of the same split, on the six s1eval carves (95% question-bootstrap interval; call by docs/SCREENS.md's rule). Bold: step 1's eleven primary reads. zs: read zero-shot.

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
|---|---|---|---|---|---|---|
| J5 | **+0.0001 WITHIN** | **+0.0008 WITHIN** | **+0.0031 WITHIN** | **-0.0063 WITHIN** | **-0.0037 WITHIN** | **-0.0141 WITHIN zs** |
| L-metaqa | **+0.0000 WITHIN zs** | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN | +0.0000 WITHIN zs |
| L-squad | -0.0013 WITHIN | **-0.0075 WITHIN zs** | -0.0011 WITHIN | +0.0018 WITHIN | +0.0019 WITHIN | +0.0054 WITHIN zs |
| L-musique | +0.0004 WITHIN | +0.0005 WITHIN | **+0.0096 GAIN zs** | -0.0013 WITHIN | -0.0046 WITHIN | -0.0089 WITHIN zs |
| L-hotpotqa | -0.0014 WITHIN | +0.0028 WITHIN | +0.0033 WITHIN | **-0.0107 LOSS zs** | -0.0044 WITHIN | +0.0142 WITHIN zs |
| L-2wiki | +0.0002 WITHIN | -0.0004 WITHIN | +0.0018 WITHIN | +0.0014 WITHIN | **+0.0026 WITHIN zs** | -0.0070 WITHIN zs |

Primary reads with a GAIN: 1 of 11. Reads with a LOSS: 1 of 36.
ADOPT when at least one primary read GAINs and none of the thirty-six LOSEs (docs/BASE_ZRC_ZKIND.md).
