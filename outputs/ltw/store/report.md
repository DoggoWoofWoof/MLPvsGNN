# LTW A1: the three-way term on the store (declared in docs/LTW_LEARNED_NEXT_HOP.md section 11)

**STORE_KEEPS**: K = 1.004 (the store keeps this share of the vector term's mean T2 gain over joint.lin; +0.122 against +0.122); the bar is 0.5.

| read | joint.lin | joint.bil | store.lin | store.bil | rrf top 5 | H1 joint.bil | H1 store.bil |
|---|---:|---:|---:|---:|---:|---:|---:|
| musique | 0.495 | 0.530 | 0.495 | 0.548 | 0.4734 | 0.5125 | 0.5141 |
| 2wiki | 0.610 | 0.860 | 0.610 | 0.852 | 0.6079 | 0.7627 | 0.7610 |
| hotpotqa | 0.723 | 0.804 | 0.723 | 0.795 | 0.6846 | 0.7930 | 0.7906 |

Fit: decision form bil; lin held-out {'musique': 0.5739130434782609, '2wiki': 0.5922619047619048, 'hotpotqa': 0.7375565610859729} (epoch 8 of 15, 1569 parameters); bil held-out {'musique': 0.8173913043478261, '2wiki': 0.8809523809523809, 'hotpotqa': 0.8009049773755657} (epoch 12 of 19, 50721 parameters)
Basis: 2wiki (60000 nodes, 0.750 of the variance on 256 axes, npz sha256 0ca4797b2c3c).
store.lin repeats joint.lin to within 0.0000 T2.
