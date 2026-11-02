# LTW: learned next-hop score (declared in docs/LTW_LEARNED_NEXT_HOP.md)

**SIGNAL**: decision score `joint.bil` T2 0.530 on musique s1eval against the best fixed score `pair` 0.360 (gain +0.170); LEARNED_PAYS True (H1 dR@5 {'musique': 0.0390635774375947, '2wiki': 0.15475906488549618, 'hotpotqa': 0.10844024307900069}).

## A: the next hop among an oracle anchor's neighbours (T2)

| read | lists | ranked | rrf | cos | anc | res | pair | joint.lin | joint.bil | musique.lin | musique.bil |
|---|---|---|---|---|---|---|---|---|---|---|---|
| musique | 915 | 923 | 0.298 | 0.326 | 0.181 | 0.314 | 0.360 | 0.495 | 0.530 | 0.481 | 0.484 |
| 2wiki | 7619 | 7619 | 0.213 | 0.220 | 0.175 | 0.111 | 0.179 | 0.610 | 0.860 | 0.451 | 0.424 |
| hotpotqa | 2799 | 2799 | 0.399 | 0.372 | 0.176 | 0.420 | 0.446 | 0.723 | 0.804 | 0.653 | 0.616 |

## H1 (label-free): R@5 against rrf's top five

| read | questions | rrf_top5 | H1_rrf | H1_cos | H1_res | H1_pair | H1_joint.lin | H1_joint.bil | H1_musique.lin | H1_musique.bil |
|---|---|---|---|---|---|---|---|---|---|---|
| musique | 2417 | 0.4734 | 0.4818 | 0.4801 | 0.4750 | 0.4849 | 0.5111 | 0.5125 | 0.5036 | 0.5018 |
| 2wiki | 12576 | 0.6079 | 0.6184 | 0.6209 | 0.5905 | 0.6094 | 0.7089 | 0.7627 | 0.6692 | 0.6648 |
| hotpotqa | 7405 | 0.6846 | 0.7196 | 0.7113 | 0.7169 | 0.7258 | 0.7770 | 0.7930 | 0.7606 | 0.7580 |

## The fits (held-out T2 at the best epoch)

- `joint`: decision form bil; lin {'musique': 0.5739130434782609, '2wiki': 0.5922619047619048, 'hotpotqa': 0.7375565610859729} (epoch 8, 1569 parameters); bil {'musique': 0.8347826086956521, '2wiki': 0.8869047619047619, 'hotpotqa': 0.8054298642533937} (epoch 7, 296481 parameters)
- `musique`: decision form bil; lin {'musique': 0.6173913043478261} (epoch 3, 1569 parameters); bil {'musique': 0.8347826086956521} (epoch 4, 296481 parameters)
