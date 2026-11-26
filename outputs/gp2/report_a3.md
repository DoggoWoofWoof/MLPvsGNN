### Phase 5 (quarter, dimension lab): tau by space
| space | keys | tau L | tau U | tau M | M status | kept |
|---|---|---|---|---|---|---|
| tw40 | 60 | 0.8783 | 0.9907 | 0.9907 | OPTIMAL | 28 |
| tw40+pr | 420 | 0.7819 | 0.9749 | 0.9869 | OPTIMAL | 195 |
| tw40+vo | 180 | 0.8317 | 0.9825 | 0.9907 | OPTIMAL | 84 |
| tw40+pr+vo | 1260 | 0.7427 | 0.8786 | 0.9683 | OPTIMAL | 534 |
| tw160 | 60 | 0.8760 | 0.9892 | 0.9907 | OPTIMAL | 28 |
| tw160+pr | 420 | 0.7804 | 0.9748 | 0.9869 | OPTIMAL | 192 |
| tw160+vo | 180 | 0.8316 | 0.9813 | 0.9907 | OPTIMAL | 84 |
| tw160+pr+vo | 1260 | 0.7396 | 0.8786 | 0.9683 | OPTIMAL | 534 |
| tw320 | 60 | 0.8776 | 0.9892 | 0.9907 | OPTIMAL | 28 |
| tw320+pr | 420 | 0.7801 | 0.9748 | 0.9869 | OPTIMAL | 191 |
| tw320+vo | 180 | 0.8316 | 0.9813 | 0.9907 | OPTIMAL | 84 |
| tw320+pr+vo | 1260 | 0.7391 | 0.8786 | 0.9683 | OPTIMAL | 544 |
| cd | 105 | 0.8759 | 0.9892 | 0.9907 | OPTIMAL | 49 |
| cd+pr | 735 | 0.7777 | 0.9748 | 0.9869 | OPTIMAL | 335 |
| cd+vo | 315 | 0.8313 | 0.9812 | 0.9907 | OPTIMAL | 147 |
| cd+pr+vo | 2205 | 0.7380 | 0.8786 | 0.9683 | OPTIMAL | 932 |
| qn | 105 | 0.8208 | 0.9866 | 0.9907 | OPTIMAL | 89 |
| qn+pr | 735 | 0.7644 | 0.9339 | 0.9869 | OPTIMAL | 360 |
| qn+vo | 315 | 0.8100 | 0.9769 | 0.9826 | OPTIMAL | 164 |
| qn+pr+vo | 2205 | 0.7335 | 0.8762 | 0.9680 | OPTIMAL | 923 |
| qs | 105 | 0.8249 | 0.9894 | 0.9907 | OPTIMAL | 49 |
| qs+pr | 735 | 0.7688 | 0.9739 | 0.9869 | OPTIMAL | 340 |
| qs+vo | 315 | 0.8132 | 0.9816 | 0.9826 | OPTIMAL | 171 |
| qs+pr+vo | 2205 | 0.7336 | 0.8738 | 0.9681 | OPTIMAL | 949 |
| pr+vo+dr | 2205 | 0.7300 | 0.8763 | 0.9680 | OPTIMAL | 989 |
| pr+vo+sr | 2205 | 0.7375 | 0.8737 | 0.9681 | OPTIMAL | 949 |
| pr+vo+ur | 2205 | 0.7294 | 0.8686 | 0.9679 | OPTIMAL | 1344 |
| pr+vo+tw | 1260 | 0.7417 | 0.8786 | 0.9683 | OPTIMAL | 534 |
| pr+vo+qov | 945 | 0.7442 | 0.8692 | 0.9683 | OPTIMAL | 413 |
| pr+vo+dir | 945 | 0.6861 | 0.8703 | 0.9683 | OPTIMAL | 480 |

### C2 (quarter): per score, the best universal caps (W1 >= W2 >= W3)
- K1 ['ur', 'dot']: W [300000, 300000, 10], worst-cell price 1.0000; per cell {'multihoprag/RK': {'reach': 1.0, 'price': 1.0}, 'musique/RK': {'reach': 0.9942, 'price': 0.952}, 'squad/RK': {'reach': 0.9939, 'price': 0.9683}, 'metaqa/SRK': {'reach': 0.9771, 'price': 0.8366}, 'webqsp/SRK': {'reach': 0.9579, 'price': 0.0556}, 'hotpotqa/RK': {'reach': 0.987, 'price': 0.5375}, 'hotpotqa/SRK': {'reach': 0.9899, 'price': 0.5316}, '2wiki/RK': {'reach': 0.9838, 'price': 0.3192}, '2wiki/SRK': {'reach': 0.9895, 'price': 0.3178}}; uniform cap each cell alone needs: {'multihoprag/RK': 300, 'musique/RK': 30000, 'squad/RK': 10000, 'metaqa/SRK': 10000, 'webqsp/SRK': 300000, 'hotpotqa/RK': 1000, 'hotpotqa/SRK': 300, '2wiki/RK': 1000, '2wiki/SRK': 300}
- K2 ['-rank', 'dot']: W [300000, 300000, 10], worst-cell price 1.0000; per cell {'multihoprag/RK': {'reach': 1.0, 'price': 1.0}, 'musique/RK': {'reach': 0.9927, 'price': 0.952}, 'squad/RK': {'reach': 0.9896, 'price': 0.9683}, 'metaqa/SRK': {'reach': 0.977, 'price': 0.8366}, 'webqsp/SRK': {'reach': 0.9517, 'price': 0.0556}, 'hotpotqa/RK': {'reach': 0.9842, 'price': 0.5375}, 'hotpotqa/SRK': {'reach': 0.9896, 'price': 0.5316}, '2wiki/RK': {'reach': 0.9815, 'price': 0.3192}, '2wiki/SRK': {'reach': 0.9868, 'price': 0.3178}}; uniform cap each cell alone needs: {'multihoprag/RK': 300, 'musique/RK': 30000, 'squad/RK': 10000, 'metaqa/SRK': 10000, 'webqsp/SRK': 300000, 'hotpotqa/RK': 30000, 'hotpotqa/SRK': 300, '2wiki/RK': 3000, '2wiki/SRK': 300}
- K3 ['ur', '-rank', 'votes', 'dot']: W [300000, 300000, 10], worst-cell price 1.0000; per cell {'multihoprag/RK': {'reach': 1.0, 'price': 1.0}, 'musique/RK': {'reach': 0.9947, 'price': 0.952}, 'squad/RK': {'reach': 0.9939, 'price': 0.9683}, 'metaqa/SRK': {'reach': 0.9771, 'price': 0.8366}, 'webqsp/SRK': {'reach': 0.9598, 'price': 0.0556}, 'hotpotqa/RK': {'reach': 0.9896, 'price': 0.5375}, 'hotpotqa/SRK': {'reach': 0.9932, 'price': 0.5316}, '2wiki/RK': {'reach': 0.988, 'price': 0.3192}, '2wiki/SRK': {'reach': 0.9924, 'price': 0.3178}}; uniform cap each cell alone needs: {'multihoprag/RK': 300, 'musique/RK': 10000, 'squad/RK': 10000, 'metaqa/SRK': 10000, 'webqsp/SRK': 300000, 'hotpotqa/RK': 1000, 'hotpotqa/SRK': 300, '2wiki/RK': 300, '2wiki/SRK': 300}
- K4 ['dot']: W ['inf', 'inf', 10], worst-cell price 1.0000; per cell {'multihoprag/RK': {'reach': 1.0, 'price': 1.0}, 'musique/RK': {'reach': 0.9921, 'price': 0.961}, 'squad/RK': {'reach': 0.9896, 'price': 0.9683}, 'metaqa/SRK': {'reach': 0.977, 'price': 0.8366}, 'webqsp/SRK': {'reach': 0.9993, 'price': 0.991}, 'hotpotqa/RK': {'reach': 0.9927, 'price': 0.9433}, 'hotpotqa/SRK': {'reach': 0.9958, 'price': 0.9157}, '2wiki/RK': {'reach': 0.9967, 'price': 0.9182}, '2wiki/SRK': {'reach': 0.9977, 'price': 0.894}}; uniform cap each cell alone needs: {'multihoprag/RK': 300, 'musique/RK': 100000, 'squad/RK': 10000, 'metaqa/SRK': 30000, 'webqsp/SRK': 'inf', 'hotpotqa/RK': 100000, 'hotpotqa/SRK': 300, '2wiki/RK': 10000, '2wiki/SRK': 300}
- K5 ['tw', 'ur', '-rank', 'dot']: W [300000, 300000, 10], worst-cell price 1.0000; per cell {'multihoprag/RK': {'reach': 1.0, 'price': 1.0}, 'musique/RK': {'reach': 0.9918, 'price': 0.952}, 'squad/RK': {'reach': 0.9921, 'price': 0.9683}, 'metaqa/SRK': {'reach': 0.9771, 'price': 0.8366}, 'webqsp/SRK': {'reach': 0.96, 'price': 0.0556}, 'hotpotqa/RK': {'reach': 0.9865, 'price': 0.5375}, 'hotpotqa/SRK': {'reach': 0.9911, 'price': 0.5316}, '2wiki/RK': {'reach': 0.9863, 'price': 0.3192}, '2wiki/SRK': {'reach': 0.9909, 'price': 0.3178}}; uniform cap each cell alone needs: {'multihoprag/RK': 1000, 'musique/RK': 10000, 'squad/RK': 3000, 'metaqa/SRK': 10000, 'webqsp/SRK': 300000, 'hotpotqa/RK': 3000, 'hotpotqa/SRK': 300, '2wiki/RK': 1000, '2wiki/SRK': 100}
- K6 ['cd', 'ur', '-rank', 'dot']: W [300000, 300000, 10], worst-cell price 1.0000; per cell {'multihoprag/RK': {'reach': 1.0, 'price': 1.0}, 'musique/RK': {'reach': 0.9918, 'price': 0.952}, 'squad/RK': {'reach': 0.9921, 'price': 0.9683}, 'metaqa/SRK': {'reach': 0.9771, 'price': 0.8366}, 'webqsp/SRK': {'reach': 0.96, 'price': 0.0556}, 'hotpotqa/RK': {'reach': 0.9859, 'price': 0.5375}, 'hotpotqa/SRK': {'reach': 0.9909, 'price': 0.5316}, '2wiki/RK': {'reach': 0.9863, 'price': 0.3192}, '2wiki/SRK': {'reach': 0.9909, 'price': 0.3178}}; uniform cap each cell alone needs: {'multihoprag/RK': 1000, 'musique/RK': 10000, 'squad/RK': 3000, 'metaqa/SRK': 10000, 'webqsp/SRK': 300000, 'hotpotqa/RK': 3000, 'hotpotqa/SRK': 300, '2wiki/RK': 1000, '2wiki/SRK': 100}
- node bound at the chosen caps: {"multihoprag/RK": {"price": 1.0, "kept_links_per_q": 589.0, "candidate_links_per_q": 589.0, "price_at_2500_links_per_q": 1.0, "S_links_per_q": 0.0}, "musique/RK": {"price": 0.952, "kept_links_per_q": 34798.1, "candidate_links_per_q": 36552.6, "price_at_2500_links_per_q": 0.06839, "S_links_per_q": 0.0}, "squad/RK": {"price": 0.9683, "kept_links_per_q": 5074.9, "candidate_links_per_q": 5241.0, "price_at_2500_links_per_q": 0.47701, "S_links_per_q": 0.0}, "metaqa/SRK": {"price": 0.8366, "kept_links_per_q": 8114.9, "candidate_links_per_q": 9699.9, "price_at_2500_links_per_q": 0.25773, "S_links_per_q": 3773.6}, "webqsp/SRK": {"price": 0.0556, "kept_links_per_q": 95551.1, "candidate_links_per_q": 1718544.7, "price_at_2500_links_per_q": 0.00145, "S_links_per_q": 28360.3}, "hotpotqa/RK": {"price": 0.5375, "kept_links_per_q": 45103.5, "candidate_links_per_q": 83913.4, "price_at_2500_links_per_q": 0.02979, "S_links_per_q": 0.0}, "hotpotqa/SRK": {"price": 0.5316, "kept_links_per_q": 23878.8, "candidate_links_per_q": 44918.8, "price_at_2500_links_per_q": 0.05566, "S_links_per_q": 3975.6}, "2wiki/RK": {"price": 0.3192, "kept_links_per_q": 49462.9, "candidate_links_per_q": 154959.1, "price_at_2500_links_per_q": 0.01613, "S_links_per_q": 0.0}, "2wiki/SRK": {"price": 0.3178, "kept_links_per_q": 37083.0, "candidate_links_per_q": 116686.7, "price_at_2500_links_per_q": 0.02142, "S_links_per_q": 3814.6}}

### C2r (quarter): per score, the best universal shares (phi1 >= phi2 >= phi3), the uniform share each cell alone needs, and the own optima (a triple per cell alone)
- K1 ['ur', 'dot']: phi [1.0, 0.9, 0.0003], worst-cell price 0.9922; per cell {'multihoprag/RK': {'reach': 1.0, 'price': 0.9922}, 'musique/RK': {'reach': 0.9846, 'price': 0.8662}, 'squad/RK': {'reach': 0.9877, 'price': 0.8804}, 'metaqa/SRK': {'reach': 0.9577, 'price': 0.7921}, 'webqsp/SRK': {'reach': 0.997, 'price': 0.894}, 'hotpotqa/RK': {'reach': 0.9911, 'price': 0.8491}, 'hotpotqa/SRK': {'reach': 0.9953, 'price': 0.8331}, '2wiki/RK': {'reach': 0.9963, 'price': 0.8264}, '2wiki/SRK': {'reach': 0.9978, 'price': 0.8079}}; uniform share each cell alone needs: {'multihoprag/RK': 0.3, 'musique/RK': 1.0, 'squad/RK': 0.9, 'metaqa/SRK': 1.0, 'webqsp/SRK': 0.7, 'hotpotqa/RK': 1.0, 'hotpotqa/SRK': 0.7, '2wiki/RK': 0.9, '2wiki/SRK': 0.7}
  - own optimum, absolute caps: {"multihoprag/RK": {"W": [300, 10, 10], "reach": 0.9746, "price": 0.478849, "kept_links_per_q": 282.0}, "musique/RK": {"W": [30000, 30000, 10], "reach": 0.9592, "price": 0.261109, "kept_links_per_q": 9544.2}, "squad/RK": {"W": [10000, 10000, 10], "reach": 0.9739, "price": 0.518151, "kept_links_per_q": 2715.6}, "metaqa/SRK": {"W": [30000, 30000, 10], "reach": 0.9771, "price": 0.83643, "kept_links_per_q": 8113.3}, "webqsp/SRK": {"W": [300000, 300000, 10], "reach": 0.9579, "price": 0.055635, "kept_links_per_q": 95611.2}, "hotpotqa/RK": {"W": [1000, 100, 10], "reach": 0.9525, "price": 0.001584, "kept_links_per_q": 132.9}, "hotpotqa/SRK": {"W": [300, 10, 10], "reach": 0.965, "price": 0.089996, "kept_links_per_q": 4042.5}, "2wiki/RK": {"W": [1000, 10, 10], "reach": 0.9529, "price": 0.000765, "kept_links_per_q": 118.5}, "2wiki/SRK": {"W": [300, 10, 10], "reach": 0.9728, "price": 0.033451, "kept_links_per_q": 3903.3}}
  - own optimum, relative caps: {"multihoprag/RK": {"W": [0.3, 0.0003, 0.0003], "reach": 0.9596, "price": 0.276651, "kept_links_per_q": 162.9}, "musique/RK": {"W": [1.0, 0.5, 0.0003], "reach": 0.954, "price": 0.486808, "kept_links_per_q": 17794.1}, "squad/RK": {"W": [0.9, 0.5, 0.03], "reach": 0.9526, "price": 0.521511, "kept_links_per_q": 2733.2}, "metaqa/SRK": {"W": [1.0, 0.9, 0.0003], "reach": 0.9577, "price": 0.792122, "kept_links_per_q": 7683.5}, "webqsp/SRK": {"W": [1.0, 0.1, 0.0003], "reach": 0.9515, "price": 0.118152, "kept_links_per_q": 203049.5}, "hotpotqa/RK": {"W": [1.0, 0.003, 0.0003], "reach": 0.9573, "price": 0.004181, "kept_links_per_q": 350.8}, "hotpotqa/SRK": {"W": [0.9, 0.0003, 0.0003], "reach": 0.9533, "price": 0.090247, "kept_links_per_q": 4053.8}, "2wiki/RK": {"W": [1.0, 0.0003, 0.0003], "reach": 0.9585, "price": 0.00106, "kept_links_per_q": 164.3}, "2wiki/SRK": {"W": [0.9, 0.0003, 0.0003], "reach": 0.9618, "price": 0.033674, "kept_links_per_q": 3929.3}}
- K2 ['-rank', 'dot']: phi [1.0, 0.7, 0.1], worst-cell price 0.9766; per cell {'multihoprag/RK': {'reach': 0.999, 'price': 0.9766}, 'musique/RK': {'reach': 0.9682, 'price': 0.6804}, 'squad/RK': {'reach': 0.974, 'price': 0.7082}, 'metaqa/SRK': {'reach': 0.9525, 'price': 0.7198}, 'webqsp/SRK': {'reach': 0.978, 'price': 0.701}, 'hotpotqa/RK': {'reach': 0.9827, 'price': 0.6664}, 'hotpotqa/SRK': {'reach': 0.9909, 'price': 0.6764}, '2wiki/RK': {'reach': 0.9856, 'price': 0.6511}, '2wiki/SRK': {'reach': 0.9902, 'price': 0.6464}}; uniform share each cell alone needs: {'multihoprag/RK': 0.7, 'musique/RK': 0.9, 'squad/RK': 1.0, 'metaqa/SRK': 0.9, 'webqsp/SRK': 0.9, 'hotpotqa/RK': 0.9, 'hotpotqa/SRK': 0.7, '2wiki/RK': 0.9, '2wiki/SRK': 0.7}
  - own optimum, absolute caps: {"multihoprag/RK": {"W": [300, 10, 10], "reach": 0.9563, "price": 0.478849, "kept_links_per_q": 282.0}, "musique/RK": {"W": [30000, 30000, 100], "reach": 0.9501, "price": 0.261394, "kept_links_per_q": 9554.6}, "squad/RK": {"W": [10000, 10000, 10], "reach": 0.9612, "price": 0.518151, "kept_links_per_q": 2715.6}, "metaqa/SRK": {"W": [10000, 10000, 3000], "reach": 0.956, "price": 0.803758, "kept_links_per_q": 7796.4}, "webqsp/SRK": {"W": [300000, 300000, 10], "reach": 0.9517, "price": 0.055635, "kept_links_per_q": 95611.2}, "hotpotqa/RK": {"W": [30000, 30000, 10], "reach": 0.9536, "price": 0.079084, "kept_links_per_q": 6636.2}, "hotpotqa/SRK": {"W": [300, 10, 10], "reach": 0.9546, "price": 0.089996, "kept_links_per_q": 4042.5}, "2wiki/RK": {"W": [3000, 3000, 300], "reach": 0.9503, "price": 0.00476, "kept_links_per_q": 737.6}, "2wiki/SRK": {"W": [300, 10, 10], "reach": 0.9565, "price": 0.033451, "kept_links_per_q": 3903.3}}
  - own optimum, relative caps: {"multihoprag/RK": {"W": [0.7, 0.0003, 0.0003], "reach": 0.9809, "price": 0.645488, "kept_links_per_q": 380.2}, "musique/RK": {"W": [1.0, 0.7, 0.0003], "reach": 0.9647, "price": 0.676491, "kept_links_per_q": 24727.5}, "squad/RK": {"W": [1.0, 0.5, 0.0003], "reach": 0.9535, "price": 0.529694, "kept_links_per_q": 2776.1}, "metaqa/SRK": {"W": [1.0, 0.7, 0.1], "reach": 0.9525, "price": 0.719831, "kept_links_per_q": 6982.3}, "webqsp/SRK": {"W": [1.0, 0.3, 0.0003], "reach": 0.9579, "price": 0.312121, "kept_links_per_q": 536393.9}, "hotpotqa/RK": {"W": [1.0, 0.2, 0.0003], "reach": 0.9504, "price": 0.189737, "kept_links_per_q": 15921.5}, "hotpotqa/SRK": {"W": [1.0, 0.0003, 0.0003], "reach": 0.9605, "price": 0.09041, "kept_links_per_q": 4061.1}, "2wiki/RK": {"W": [1.0, 0.03, 0.0003], "reach": 0.9516, "price": 0.028307, "kept_links_per_q": 4386.4}, "2wiki/SRK": {"W": [1.0, 0.0003, 0.0003], "reach": 0.962, "price": 0.033751, "kept_links_per_q": 3938.3}}
- K3 ['ur', '-rank', 'votes', 'dot']: phi [0.9, 0.9, 0.5], worst-cell price 0.9000; per cell {'multihoprag/RK': {'reach': 0.999, 'price': 0.9}, 'musique/RK': {'reach': 0.9614, 'price': 0.8844}, 'squad/RK': {'reach': 0.9757, 'price': 0.8872}, 'metaqa/SRK': {'reach': 0.9524, 'price': 0.8735}, 'webqsp/SRK': {'reach': 0.9779, 'price': 0.8981}, 'hotpotqa/RK': {'reach': 0.9669, 'price': 0.8773}, 'hotpotqa/SRK': {'reach': 0.9886, 'price': 0.8751}, '2wiki/RK': {'reach': 0.9783, 'price': 0.8673}, '2wiki/SRK': {'reach': 0.9867, 'price': 0.8609}}; uniform share each cell alone needs: {'multihoprag/RK': 0.3, 'musique/RK': 0.9, 'squad/RK': 0.9, 'metaqa/SRK': 0.9, 'webqsp/SRK': 0.9, 'hotpotqa/RK': 0.9, 'hotpotqa/SRK': 0.7, '2wiki/RK': 0.9, '2wiki/SRK': 0.7}
  - own optimum, absolute caps: {"multihoprag/RK": {"W": [300, 10, 10], "reach": 0.9789, "price": 0.478849, "kept_links_per_q": 282.0}, "musique/RK": {"W": [10000, 10000, 30], "reach": 0.9512, "price": 0.120582, "kept_links_per_q": 4407.6}, "squad/RK": {"W": [10000, 10000, 10], "reach": 0.9783, "price": 0.518151, "kept_links_per_q": 2715.6}, "metaqa/SRK": {"W": [10000, 10000, 300], "reach": 0.9515, "price": 0.765525, "kept_links_per_q": 7425.5}, "webqsp/SRK": {"W": [300000, 300000, 10], "reach": 0.9598, "price": 0.055635, "kept_links_per_q": 95611.2}, "hotpotqa/RK": {"W": [1000, 100, 10], "reach": 0.9528, "price": 0.001584, "kept_links_per_q": 132.9}, "hotpotqa/SRK": {"W": [300, 10, 10], "reach": 0.9694, "price": 0.089996, "kept_links_per_q": 4042.5}, "2wiki/RK": {"W": [1000, 10, 10], "reach": 0.9557, "price": 0.000765, "kept_links_per_q": 118.5}, "2wiki/SRK": {"W": [300, 10, 10], "reach": 0.9736, "price": 0.033451, "kept_links_per_q": 3903.3}}
  - own optimum, relative caps: {"multihoprag/RK": {"W": [0.3, 0.0003, 0.0003], "reach": 0.9662, "price": 0.276651, "kept_links_per_q": 162.9}, "musique/RK": {"W": [1.0, 0.3, 0.01], "reach": 0.9519, "price": 0.297503, "kept_links_per_q": 10874.5}, "squad/RK": {"W": [1.0, 0.3, 0.03], "reach": 0.9535, "price": 0.355293, "kept_links_per_q": 1862.1}, "metaqa/SRK": {"W": [1.0, 0.7, 0.03], "reach": 0.9508, "price": 0.708377, "kept_links_per_q": 6871.2}, "webqsp/SRK": {"W": [1.0, 0.1, 0.0003], "reach": 0.9503, "price": 0.118152, "kept_links_per_q": 203049.5}, "hotpotqa/RK": {"W": [1.0, 0.001, 0.0003], "reach": 0.9509, "price": 0.002297, "kept_links_per_q": 192.7}, "hotpotqa/SRK": {"W": [0.9, 0.0003, 0.0003], "reach": 0.9658, "price": 0.090247, "kept_links_per_q": 4053.8}, "2wiki/RK": {"W": [1.0, 0.0003, 0.0003], "reach": 0.9594, "price": 0.00106, "kept_links_per_q": 164.3}, "2wiki/SRK": {"W": [0.9, 0.0003, 0.0003], "reach": 0.9671, "price": 0.033674, "kept_links_per_q": 3929.3}}
- K4 ['dot']: phi [1.0, 0.9, 0.0003], worst-cell price 0.9922; per cell {'multihoprag/RK': {'reach': 1.0, 'price': 0.9922}, 'musique/RK': {'reach': 0.9824, 'price': 0.8662}, 'squad/RK': {'reach': 0.9818, 'price': 0.8804}, 'metaqa/SRK': {'reach': 0.9526, 'price': 0.7921}, 'webqsp/SRK': {'reach': 0.9958, 'price': 0.894}, 'hotpotqa/RK': {'reach': 0.9847, 'price': 0.8491}, 'hotpotqa/SRK': {'reach': 0.9911, 'price': 0.8331}, '2wiki/RK': {'reach': 0.9925, 'price': 0.8264}, '2wiki/SRK': {'reach': 0.9949, 'price': 0.8079}}; uniform share each cell alone needs: {'multihoprag/RK': 0.7, 'musique/RK': 1.0, 'squad/RK': 0.9, 'metaqa/SRK': 1.0, 'webqsp/SRK': 0.7, 'hotpotqa/RK': 1.0, 'hotpotqa/SRK': 0.9, '2wiki/RK': 0.9, '2wiki/SRK': 0.9}
  - own optimum, absolute caps: {"multihoprag/RK": {"W": [300, 10, 10], "reach": 0.9679, "price": 0.478849, "kept_links_per_q": 282.0}, "musique/RK": {"W": [100000, 100000, 10], "reach": 0.9675, "price": 0.647412, "kept_links_per_q": 23664.6}, "squad/RK": {"W": [10000, 10000, 10], "reach": 0.9567, "price": 0.518151, "kept_links_per_q": 2715.6}, "metaqa/SRK": {"W": [30000, 30000, 10], "reach": 0.977, "price": 0.83643, "kept_links_per_q": 8113.3}, "webqsp/SRK": {"W": ["inf", "inf", 10], "reach": 0.9993, "price": 0.991008, "kept_links_per_q": 1703091.5}, "hotpotqa/RK": {"W": [100000, 100000, 10], "reach": 0.9661, "price": 0.233213, "kept_links_per_q": 19569.7}, "hotpotqa/SRK": {"W": [300, 10, 10], "reach": 0.9502, "price": 0.089996, "kept_links_per_q": 4042.5}, "2wiki/RK": {"W": [10000, 10000, 10], "reach": 0.9509, "price": 0.013935, "kept_links_per_q": 2159.4}, "2wiki/SRK": {"W": [300, 10, 10], "reach": 0.9558, "price": 0.033451, "kept_links_per_q": 3903.3}}
  - own optimum, relative caps: {"multihoprag/RK": {"W": [0.7, 0.0003, 0.0003], "reach": 0.9811, "price": 0.645488, "kept_links_per_q": 380.2}, "musique/RK": {"W": [1.0, 0.7, 0.0003], "reach": 0.9575, "price": 0.676491, "kept_links_per_q": 24727.5}, "squad/RK": {"W": [1.0, 0.5, 0.0003], "reach": 0.9552, "price": 0.529694, "kept_links_per_q": 2776.1}, "metaqa/SRK": {"W": [1.0, 0.9, 0.0003], "reach": 0.9526, "price": 0.792122, "kept_links_per_q": 7683.5}, "webqsp/SRK": {"W": [1.0, 0.2, 0.0003], "reach": 0.9528, "price": 0.215136, "kept_links_per_q": 369720.8}, "hotpotqa/RK": {"W": [1.0, 0.3, 0.1], "reach": 0.9512, "price": 0.289587, "kept_links_per_q": 24300.2}, "hotpotqa/SRK": {"W": [1.0, 0.0003, 0.0003], "reach": 0.9572, "price": 0.09041, "kept_links_per_q": 4061.1}, "2wiki/RK": {"W": [1.0, 0.1, 0.0003], "reach": 0.9552, "price": 0.092525, "kept_links_per_q": 14337.6}, "2wiki/SRK": {"W": [1.0, 0.0003, 0.0003], "reach": 0.9599, "price": 0.033751, "kept_links_per_q": 3938.3}}
- K5 ['tw', 'ur', '-rank', 'dot']: phi [0.9, 0.9, 0.5], worst-cell price 0.9000; per cell {'multihoprag/RK': {'reach': 0.9987, 'price': 0.9}, 'musique/RK': {'reach': 0.9666, 'price': 0.8844}, 'squad/RK': {'reach': 0.9875, 'price': 0.8872}, 'metaqa/SRK': {'reach': 0.9524, 'price': 0.8735}, 'webqsp/SRK': {'reach': 0.9793, 'price': 0.8981}, 'hotpotqa/RK': {'reach': 0.9958, 'price': 0.8773}, 'hotpotqa/SRK': {'reach': 0.999, 'price': 0.8751}, '2wiki/RK': {'reach': 0.9977, 'price': 0.8673}, '2wiki/SRK': {'reach': 0.9986, 'price': 0.8609}}; uniform share each cell alone needs: {'multihoprag/RK': 0.5, 'musique/RK': 0.9, 'squad/RK': 0.7, 'metaqa/SRK': 0.9, 'webqsp/SRK': 0.7, 'hotpotqa/RK': 0.7, 'hotpotqa/SRK': 0.5, '2wiki/RK': 0.7, '2wiki/SRK': 0.5}
  - own optimum, absolute caps: {"multihoprag/RK": {"W": [1000, 10, 10], "reach": 0.9934, "price": 0.922577, "kept_links_per_q": 543.4}, "musique/RK": {"W": [10000, 10000, 1000], "reach": 0.9506, "price": 0.123017, "kept_links_per_q": 4496.6}, "squad/RK": {"W": [3000, 3000, 100], "reach": 0.9511, "price": 0.230201, "kept_links_per_q": 1206.5}, "metaqa/SRK": {"W": [10000, 10000, 300], "reach": 0.9511, "price": 0.765525, "kept_links_per_q": 7425.5}, "webqsp/SRK": {"W": [300000, 300000, 10], "reach": 0.96, "price": 0.055635, "kept_links_per_q": 95611.2}, "hotpotqa/RK": {"W": [3000, 3000, 10], "reach": 0.9548, "price": 0.009646, "kept_links_per_q": 809.4}, "hotpotqa/SRK": {"W": [300, 10, 10], "reach": 0.9599, "price": 0.089996, "kept_links_per_q": 4042.5}, "2wiki/RK": {"W": [1000, 1000, 10], "reach": 0.9559, "price": 0.002071, "kept_links_per_q": 320.9}, "2wiki/SRK": {"W": [100, 10, 10], "reach": 0.9522, "price": 0.033247, "kept_links_per_q": 3879.5}}
  - own optimum, relative caps: {"multihoprag/RK": {"W": [0.5, 0.0003, 0.0003], "reach": 0.978, "price": 0.461069, "kept_links_per_q": 271.6}, "musique/RK": {"W": [1.0, 0.5, 0.0003], "reach": 0.9563, "price": 0.486808, "kept_links_per_q": 17794.1}, "squad/RK": {"W": [0.9, 0.5, 0.0003], "reach": 0.9538, "price": 0.520563, "kept_links_per_q": 2728.3}, "metaqa/SRK": {"W": [1.0, 0.7, 0.03], "reach": 0.9501, "price": 0.708377, "kept_links_per_q": 6871.2}, "webqsp/SRK": {"W": [1.0, 0.1, 0.0003], "reach": 0.9503, "price": 0.118152, "kept_links_per_q": 203049.5}, "hotpotqa/RK": {"W": [0.9, 0.03, 0.0003], "reach": 0.9616, "price": 0.029478, "kept_links_per_q": 2473.6}, "hotpotqa/SRK": {"W": [0.7, 0.0003, 0.0003], "reach": 0.9573, "price": 0.089921, "kept_links_per_q": 4039.1}, "2wiki/RK": {"W": [0.9, 0.003, 0.0003], "reach": 0.9524, "price": 0.003461, "kept_links_per_q": 536.3}, "2wiki/SRK": {"W": [0.5, 0.0003, 0.0003], "reach": 0.951, "price": 0.033366, "kept_links_per_q": 3893.4}}
- K6 ['cd', 'ur', '-rank', 'dot']: phi [0.9, 0.9, 0.5], worst-cell price 0.9000; per cell {'multihoprag/RK': {'reach': 0.9987, 'price': 0.9}, 'musique/RK': {'reach': 0.967, 'price': 0.8844}, 'squad/RK': {'reach': 0.9875, 'price': 0.8872}, 'metaqa/SRK': {'reach': 0.9524, 'price': 0.8735}, 'webqsp/SRK': {'reach': 0.9793, 'price': 0.8981}, 'hotpotqa/RK': {'reach': 0.9928, 'price': 0.8773}, 'hotpotqa/SRK': {'reach': 0.999, 'price': 0.8751}, '2wiki/RK': {'reach': 0.9977, 'price': 0.8673}, '2wiki/SRK': {'reach': 0.9986, 'price': 0.8609}}; uniform share each cell alone needs: {'multihoprag/RK': 0.5, 'musique/RK': 0.9, 'squad/RK': 0.7, 'metaqa/SRK': 0.9, 'webqsp/SRK': 0.7, 'hotpotqa/RK': 0.7, 'hotpotqa/SRK': 0.5, '2wiki/RK': 0.7, '2wiki/SRK': 0.5}
  - own optimum, absolute caps: {"multihoprag/RK": {"W": [1000, 10, 10], "reach": 0.9934, "price": 0.922577, "kept_links_per_q": 543.4}, "musique/RK": {"W": [10000, 10000, 1000], "reach": 0.9501, "price": 0.123017, "kept_links_per_q": 4496.6}, "squad/RK": {"W": [3000, 3000, 100], "reach": 0.9511, "price": 0.230201, "kept_links_per_q": 1206.5}, "metaqa/SRK": {"W": [10000, 10000, 300], "reach": 0.9511, "price": 0.765525, "kept_links_per_q": 7425.5}, "webqsp/SRK": {"W": [300000, 300000, 10], "reach": 0.96, "price": 0.055635, "kept_links_per_q": 95611.2}, "hotpotqa/RK": {"W": [3000, 3000, 10], "reach": 0.9536, "price": 0.009646, "kept_links_per_q": 809.4}, "hotpotqa/SRK": {"W": [300, 10, 10], "reach": 0.9599, "price": 0.089996, "kept_links_per_q": 4042.5}, "2wiki/RK": {"W": [1000, 1000, 10], "reach": 0.9553, "price": 0.002071, "kept_links_per_q": 320.9}, "2wiki/SRK": {"W": [100, 10, 10], "reach": 0.9526, "price": 0.033247, "kept_links_per_q": 3879.5}}
  - own optimum, relative caps: {"multihoprag/RK": {"W": [0.5, 0.0003, 0.0003], "reach": 0.978, "price": 0.461069, "kept_links_per_q": 271.6}, "musique/RK": {"W": [1.0, 0.5, 0.0003], "reach": 0.9563, "price": 0.486808, "kept_links_per_q": 17794.1}, "squad/RK": {"W": [0.9, 0.5, 0.0003], "reach": 0.9538, "price": 0.520563, "kept_links_per_q": 2728.3}, "metaqa/SRK": {"W": [1.0, 0.7, 0.03], "reach": 0.9501, "price": 0.708377, "kept_links_per_q": 6871.2}, "webqsp/SRK": {"W": [1.0, 0.1, 0.0003], "reach": 0.9503, "price": 0.118152, "kept_links_per_q": 203049.5}, "hotpotqa/RK": {"W": [0.9, 0.03, 0.0003], "reach": 0.9575, "price": 0.029478, "kept_links_per_q": 2473.6}, "hotpotqa/SRK": {"W": [0.7, 0.0003, 0.0003], "reach": 0.9563, "price": 0.089921, "kept_links_per_q": 4039.1}, "2wiki/RK": {"W": [0.9, 0.003, 0.0003], "reach": 0.9519, "price": 0.003461, "kept_links_per_q": 536.3}, "2wiki/SRK": {"W": [0.7, 0.0003, 0.0003], "reach": 0.9641, "price": 0.03352, "kept_links_per_q": 3911.3}}
- node bound at the chosen shares: {"multihoprag/RK": {"price": 0.9, "kept_links_per_q": 530.1, "candidate_links_per_q": 589.0, "price_at_2500_links_per_q": 1.0, "S_links_per_q": 0.0}, "musique/RK": {"price": 0.8844, "kept_links_per_q": 32327.1, "candidate_links_per_q": 36552.6, "price_at_2500_links_per_q": 0.06839, "S_links_per_q": 0.0}, "squad/RK": {"price": 0.8872, "kept_links_per_q": 4649.8, "candidate_links_per_q": 5241.0, "price_at_2500_links_per_q": 0.47701, "S_links_per_q": 0.0}, "metaqa/SRK": {"price": 0.8735, "kept_links_per_q": 8472.9, "candidate_links_per_q": 9699.9, "price_at_2500_links_per_q": 0.25773, "S_links_per_q": 3773.6}, "webqsp/SRK": {"price": 0.8981, "kept_links_per_q": 1543425.0, "candidate_links_per_q": 1718544.7, "price_at_2500_links_per_q": 0.00145, "S_links_per_q": 28360.3}, "hotpotqa/RK": {"price": 0.8773, "kept_links_per_q": 73617.2, "candidate_links_per_q": 83913.4, "price_at_2500_links_per_q": 0.02979, "S_links_per_q": 0.0}, "hotpotqa/SRK": {"price": 0.8751, "kept_links_per_q": 39308.4, "candidate_links_per_q": 44918.8, "price_at_2500_links_per_q": 0.05566, "S_links_per_q": 3975.6}, "2wiki/RK": {"price": 0.8673, "kept_links_per_q": 134396.0, "candidate_links_per_q": 154959.1, "price_at_2500_links_per_q": 0.01613, "S_links_per_q": 0.0}, "2wiki/SRK": {"price": 0.8609, "kept_links_per_q": 100455.6, "candidate_links_per_q": 116686.7, "price_at_2500_links_per_q": 0.02142, "S_links_per_q": 3814.6}}

### Final on the quarter: C2r K3 tau 0.9000 (rule rule_C2r; record phase a4); section 13's best was lab:pr+vo 0.9683

**multihoprag/RK** (quarter, per rule: reach / price / kept links per question / candidates per question / price at 2,500 / S links per question)
  - rule_s11_final: 0.9959 / 0.9953 / 586.2 / 589.0 / 1.0 / 0.0
  - rule_A: 0.988 / 0.9953 / 586.2 / 589.0 / 1.0 / 0.0
  - rule_B1: 0.988 / 0.9953 / 586.2 / 589.0 / 1.0 / 0.0
  - rule_C: 1.0 / 0.9996 / 588.7 / 589.0 / 1.0 / 0.0
  - rule_A2: 0.9895 / 0.842 / 495.9 / 589.0 / 1.0 / 0.0
  - rule_P5: 0.9997 / 0.939 / 553.1 / 589.0 / 1.0 / 0.0
  - rule_C2: 1.0 / 1.0 / 589.0 / 589.0 / 1.0 / 0.0
  - rule_C2r: 0.999 / 0.9 / 530.1 / 589.0 / 1.0 / 0.0

**musique/RK** (quarter, per rule: reach / price / kept links per question / candidates per question / price at 2,500 / S links per question)
  - rule_s11_final: 0.95 / 0.9623 / 35175.9 / 36552.6 / 0.06839 / 0.0
  - rule_A: 0.95 / 0.9626 / 35184.5 / 36552.6 / 0.06839 / 0.0
  - rule_B1: 0.95 / 0.9615 / 35146.7 / 36552.6 / 0.06839 / 0.0
  - rule_C: 0.9808 / 0.9603 / 35100.7 / 36552.6 / 0.06839 / 0.0
  - rule_A2: 0.9574 / 0.9407 / 34384.2 / 36552.6 / 0.06839 / 0.0
  - rule_P5: 0.9507 / 0.9391 / 34327.3 / 36552.6 / 0.06839 / 0.0
  - rule_C2: 0.9942 / 0.952 / 34798.9 / 36552.6 / 0.06839 / 0.0
  - rule_C2r: 0.9614 / 0.8844 / 32327.2 / 36552.6 / 0.06839 / 0.0

**squad/RK** (quarter, per rule: reach / price / kept links per question / candidates per question / price at 2,500 / S links per question)
  - rule_s11_final: 0.9567 / 0.9819 / 5146.4 / 5241.0 / 0.47701 / 0.0
  - rule_A: 0.9597 / 0.9825 / 5149.5 / 5241.0 / 0.47701 / 0.0
  - rule_B1: 0.9595 / 0.9684 / 5075.1 / 5241.0 / 0.47701 / 0.0
  - rule_C: 0.9864 / 0.9667 / 5066.3 / 5241.0 / 0.47701 / 0.0
  - rule_A2: 0.9519 / 0.9282 / 4864.7 / 5241.0 / 0.47701 / 0.0
  - rule_P5: 0.9785 / 0.9282 / 4864.8 / 5241.0 / 0.47701 / 0.0
  - rule_C2: 0.9939 / 0.9683 / 5074.7 / 5241.0 / 0.47701 / 0.0
  - rule_C2r: 0.9757 / 0.8872 / 4649.9 / 5241.0 / 0.47701 / 0.0

**metaqa/SRK** (quarter, per rule: reach / price / kept links per question / candidates per question / price at 2,500 / S links per question)
  - rule_s11_final: 0.9646 / 0.8636 / 8377.1 / 9699.9 / 0.25773 / 3773.6
  - rule_A: 0.9646 / 0.8678 / 8417.8 / 9699.9 / 0.25773 / 3773.6
  - rule_B1: 0.9541 / 0.8327 / 8076.7 / 9699.9 / 0.25773 / 3773.6
  - rule_C: 0.9698 / 0.8251 / 8003.8 / 9699.9 / 0.25773 / 3773.6
  - rule_A2: 0.9504 / 0.6913 / 6705.5 / 9699.9 / 0.25773 / 3773.6
  - rule_P5: 0.9501 / 0.6818 / 6613.7 / 9699.9 / 0.25773 / 3773.6
  - rule_C2: 0.9771 / 0.8366 / 8114.6 / 9699.9 / 0.25773 / 3773.6
  - rule_C2r: 0.9524 / 0.8735 / 8472.5 / 9699.9 / 0.25773 / 3773.6

**webqsp/SRK** (quarter, per rule: reach / price / kept links per question / candidates per question / price at 2,500 / S links per question)
  - rule_s11_final: 0.9863 / 0.9923 / 1705229.6 / 1718544.7 / 0.00145 / 28360.3
  - rule_A: 0.9781 / 0.9923 / 1705235.6 / 1718544.7 / 0.00145 / 28360.3
  - rule_B1: 0.9784 / 0.9907 / 1702541.2 / 1718544.7 / 0.00145 / 28360.3
  - rule_C: 0.9972 / 0.9907 / 1702533.9 / 1718544.7 / 0.00145 / 28360.3
  - rule_A2: 0.9771 / 0.9683 / 1663998.1 / 1718544.7 / 0.00145 / 28360.3
  - rule_P5: 0.9764 / 0.9679 / 1663333.0 / 1718544.7 / 0.00145 / 28360.3
  - rule_C2: 0.9579 / 0.0556 / 95611.7 / 1718544.7 / 0.00145 / 28360.3
  - rule_C2r: 0.9779 / 0.8981 / 1543344.6 / 1718544.7 / 0.00145 / 28360.3

**hotpotqa/RK** (quarter, per rule: reach / price / kept links per question / candidates per question / price at 2,500 / S links per question)
  - rule_s11_final: 0.9831 / 0.9433 / 79151.9 / 83913.4 / 0.02979 / 0.0
  - rule_A: 0.9714 / 0.9433 / 79151.5 / 83913.4 / 0.02979 / 0.0
  - rule_B1: 0.9708 / 0.9432 / 79143.0 / 83913.4 / 0.02979 / 0.0
  - rule_C: 0.9865 / 0.9432 / 79147.8 / 83913.4 / 0.02979 / 0.0
  - rule_A2: 0.9504 / 0.1478 / 12400.6 / 83913.4 / 0.02979 / 0.0
  - rule_P5: 0.9518 / 0.1464 / 12288.4 / 83913.4 / 0.02979 / 0.0
  - rule_C2: 0.987 / 0.5375 / 45099.7 / 83913.4 / 0.02979 / 0.0
  - rule_C2r: 0.9669 / 0.8773 / 73617.2 / 83913.4 / 0.02979 / 0.0

**hotpotqa/SRK** (quarter, per rule: reach / price / kept links per question / candidates per question / price at 2,500 / S links per question)
  - rule_s11_final: 0.9897 / 0.9157 / 41131.2 / 44918.8 / 0.05566 / 3975.6
  - rule_A: 0.9929 / 0.9157 / 41130.9 / 44918.8 / 0.05566 / 3975.6
  - rule_B1: 0.9933 / 0.9155 / 41124.5 / 44918.8 / 0.05566 / 3975.6
  - rule_C: 0.9924 / 0.9156 / 41128.7 / 44918.8 / 0.05566 / 3975.6
  - rule_A2: 0.9749 / 0.217 / 9749.4 / 44918.8 / 0.05566 / 3975.6
  - rule_P5: 0.9778 / 0.2158 / 9695.5 / 44918.8 / 0.05566 / 3975.6
  - rule_C2: 0.9899 / 0.5316 / 23879.6 / 44918.8 / 0.05566 / 3975.6
  - rule_C2r: 0.9886 / 0.8751 / 39309.3 / 44918.8 / 0.05566 / 3975.6

**2wiki/RK** (quarter, per rule: reach / price / kept links per question / candidates per question / price at 2,500 / S links per question)
  - rule_s11_final: 0.9801 / 0.9182 / 142281.1 / 154959.1 / 0.01613 / 0.0
  - rule_A: 0.9791 / 0.9182 / 142280.9 / 154959.1 / 0.01613 / 0.0
  - rule_B1: 0.9819 / 0.9182 / 142277.5 / 154959.1 / 0.01613 / 0.0
  - rule_C: 0.9949 / 0.9181 / 142274.5 / 154959.1 / 0.01613 / 0.0
  - rule_A2: 0.9611 / 0.3109 / 48169.8 / 154959.1 / 0.01613 / 0.0
  - rule_P5: 0.9685 / 0.3102 / 48064.3 / 154959.1 / 0.01613 / 0.0
  - rule_C2: 0.9838 / 0.3192 / 49457.0 / 154959.1 / 0.01613 / 0.0
  - rule_C2r: 0.9783 / 0.8673 / 134390.5 / 154959.1 / 0.01613 / 0.0

**2wiki/SRK** (quarter, per rule: reach / price / kept links per question / candidates per question / price at 2,500 / S links per question)
  - rule_s11_final: 0.9861 / 0.894 / 104315.4 / 116686.7 / 0.02142 / 3814.6
  - rule_A: 0.9851 / 0.894 / 104315.3 / 116686.7 / 0.02142 / 3814.6
  - rule_B1: 0.9878 / 0.8939 / 104310.2 / 116686.7 / 0.02142 / 3814.6
  - rule_C: 0.9961 / 0.8939 / 104310.1 / 116686.7 / 0.02142 / 3814.6
  - rule_A2: 0.9779 / 0.3403 / 39707.8 / 116686.7 / 0.02142 / 3814.6
  - rule_P5: 0.9808 / 0.3397 / 39641.4 / 116686.7 / 0.02142 / 3814.6
  - rule_C2: 0.9895 / 0.3178 / 37084.7 / 116686.7 / 0.02142 / 3814.6
  - rule_C2r: 0.9867 / 0.8609 / 100449.8 / 116686.7 / 0.02142 / 3814.6

### Grade (rest of dev)
- rule_s11_final (None; tw+dr+pr): NOT_UNIVERSAL (0/9 cells SCREEN_IN)
  - multihoprag/RK: reach 0.9991, price 0.9943, path retention 1.0 (121 golds), kept links per question 552.6 of 555.7 (price at 2,500: 1.0; S per question 0.0) -> False
  - musique/RK: reach 0.9486, price 0.9313, path retention 0.9571 (1469 golds), kept links per question 30358.4 of 32599.3 (price at 2,500: 0.07669; S per question 0.0) -> False
  - squad/RK: reach 0.9326, price 0.927, path retention 0.9481 (77 golds), kept links per question 1854.0 of 2000.0 (price at 2,500: 1.0; S per question 0.0) -> False
  - metaqa/SRK: reach 0.9614, price 0.8681, path retention 0.9815 (8105 golds), kept links per question 8171.7 of 9413.0 (price at 2,500: 0.26559; S per question 3642.1) -> False
  - webqsp/SRK: reach 0.9904, price 0.9965, path retention 0.9947 (6632 golds), kept links per question 2234391.1 of 2242206.9 (price at 2,500: 0.00111; S per question 36939.0) -> False
  - hotpotqa/RK: reach 0.9672, price 0.9573, path retention 0.9798 (594 golds), kept links per question 55719.2 of 58203.9 (price at 2,500: 0.04295; S per question 0.0) -> False
  - hotpotqa/SRK: reach 0.9821, price 0.9685, path retention 0.9933 (595 golds), kept links per question 23995.0 of 24775.2 (price at 2,500: 0.10091; S per question 2945.5) -> False
  - 2wiki/RK: reach 0.9867, price 0.9336, path retention 0.9929 (1120 golds), kept links per question 169001.9 of 181014.9 (price at 2,500: 0.01381; S per question 0.0) -> False
  - 2wiki/SRK: reach 0.9887, price 0.921, path retention 0.9946 (1120 golds), kept links per question 141479.4 of 153619.6 (price at 2,500: 0.01627; S per question 6344.4) -> False
- rule_A (A (phase 1 best); dr+pr): NOT_UNIVERSAL (0/9 cells SCREEN_IN)
  - multihoprag/RK: reach 0.9965, price 0.9943, path retention 1.0 (121 golds), kept links per question 552.6 of 555.7 (price at 2,500: 1.0; S per question 0.0) -> False
  - musique/RK: reach 0.9563, price 0.9315, path retention 0.9632 (1469 golds), kept links per question 30366.7 of 32599.3 (price at 2,500: 0.07669; S per question 0.0) -> False
  - squad/RK: reach 0.9401, price 0.9317, path retention 0.9481 (77 golds), kept links per question 1863.5 of 2000.0 (price at 2,500: 1.0; S per question 0.0) -> False
  - metaqa/SRK: reach 0.9622, price 0.8717, path retention 0.9815 (8105 golds), kept links per question 8205.6 of 9413.0 (price at 2,500: 0.26559; S per question 3642.1) -> False
  - webqsp/SRK: reach 0.9853, price 0.9965, path retention 0.9919 (6632 golds), kept links per question 2234390.2 of 2242206.9 (price at 2,500: 0.00111; S per question 36939.0) -> False
  - hotpotqa/RK: reach 0.9611, price 0.9573, path retention 0.9731 (594 golds), kept links per question 55718.9 of 58203.9 (price at 2,500: 0.04295; S per question 0.0) -> False
  - hotpotqa/SRK: reach 0.9839, price 0.9685, path retention 0.995 (595 golds), kept links per question 23994.8 of 24775.2 (price at 2,500: 0.10091; S per question 2945.5) -> False
  - 2wiki/RK: reach 0.9864, price 0.9336, path retention 0.9938 (1120 golds), kept links per question 169001.6 of 181014.9 (price at 2,500: 0.01381; S per question 0.0) -> False
  - 2wiki/SRK: reach 0.9883, price 0.921, path retention 0.9955 (1120 golds), kept links per question 141479.1 of 153619.6 (price at 2,500: 0.01627; S per question 6344.4) -> False
- rule_B1 (B1 (phase 2 best); tw+dr): NOT_UNIVERSAL (0/9 cells SCREEN_IN)
  - multihoprag/RK: reach 0.9967, price 0.9944, path retention 1.0 (121 golds), kept links per question 552.6 of 555.7 (price at 2,500: 1.0; S per question 0.0) -> False
  - musique/RK: reach 0.9583, price 0.9304, path retention 0.9496 (1469 golds), kept links per question 30330.4 of 32599.3 (price at 2,500: 0.07669; S per question 0.0) -> False
  - squad/RK: reach 0.9359, price 0.8975, path retention 0.9221 (77 golds), kept links per question 1795.1 of 2000.0 (price at 2,500: 1.0; S per question 0.0) -> False
  - metaqa/SRK: reach 0.9517, price 0.8379, path retention 0.981 (8105 golds), kept links per question 7887.0 of 9413.0 (price at 2,500: 0.26559; S per question 3642.1) -> False
  - webqsp/SRK: reach 0.9854, price 0.9965, path retention 0.9916 (6632 golds), kept links per question 2234308.8 of 2242206.9 (price at 2,500: 0.00111; S per question 36939.0) -> False
  - hotpotqa/RK: reach 0.9596, price 0.9572, path retention 0.963 (594 golds), kept links per question 55713.4 of 58203.9 (price at 2,500: 0.04295; S per question 0.0) -> False
  - hotpotqa/SRK: reach 0.9839, price 0.9684, path retention 0.995 (595 golds), kept links per question 23993.1 of 24775.2 (price at 2,500: 0.10091; S per question 2945.5) -> False
  - 2wiki/RK: reach 0.9858, price 0.9336, path retention 0.9821 (1120 golds), kept links per question 168995.5 of 181014.9 (price at 2,500: 0.01381; S per question 0.0) -> False
  - 2wiki/SRK: reach 0.9878, price 0.9209, path retention 0.9955 (1120 golds), kept links per question 141472.7 of 153619.6 (price at 2,500: 0.01627; S per question 6344.4) -> False
- rule_C (C (groups x hop); S0): NOT_UNIVERSAL (0/9 cells SCREEN_IN)
  - multihoprag/RK: reach 0.9992, price 0.9991, path retention 1.0 (121 golds), kept links per question 555.3 of 555.7 (price at 2,500: 1.0; S per question 0.0) -> False
  - musique/RK: reach 0.9755, price 0.9287, path retention 0.8734 (1469 golds), kept links per question 30274.4 of 32599.3 (price at 2,500: 0.07669; S per question 0.0) -> False
  - squad/RK: reach 0.9477, price 0.8841, path retention 0.7922 (77 golds), kept links per question 1768.2 of 2000.0 (price at 2,500: 1.0; S per question 0.0) -> False
  - metaqa/SRK: reach 0.9699, price 0.8315, path retention 0.9928 (8105 golds), kept links per question 7827.2 of 9413.0 (price at 2,500: 0.26559; S per question 3642.1) -> False
  - webqsp/SRK: reach 0.999, price 0.9965, path retention 0.9985 (6632 golds), kept links per question 2234320.3 of 2242206.9 (price at 2,500: 0.00111; S per question 36939.0) -> False
  - hotpotqa/RK: reach 0.9911, price 0.9573, path retention 0.9714 (594 golds), kept links per question 55718.2 of 58203.9 (price at 2,500: 0.04295; S per question 0.0) -> False
  - hotpotqa/SRK: reach 0.997, price 0.9686, path retention 0.9916 (595 golds), kept links per question 23998.0 of 24775.2 (price at 2,500: 0.10091; S per question 2945.5) -> False
  - 2wiki/RK: reach 0.9935, price 0.9336, path retention 0.967 (1120 golds), kept links per question 168994.0 of 181014.9 (price at 2,500: 0.01381; S per question 0.0) -> False
  - 2wiki/SRK: reach 0.9954, price 0.9209, path retention 0.9973 (1120 golds), kept links per question 141472.4 of 153619.6 (price at 2,500: 0.01627; S per question 6344.4) -> False
- rule_A2 (A2 (labels); pr+vo): NOT_UNIVERSAL (4/9 cells SCREEN_IN)
  - multihoprag/RK: reach 0.9713, price 0.8412, path retention 0.9835 (121 golds), kept links per question 467.5 of 555.7 (price at 2,500: 1.0; S per question 0.0) -> False
  - musique/RK: reach 0.9505, price 0.9104, path retention 0.9319 (1469 golds), kept links per question 29677.6 of 32599.3 (price at 2,500: 0.07669; S per question 0.0) -> False
  - squad/RK: reach 0.8725, price 0.814, path retention 0.8701 (77 golds), kept links per question 1628.0 of 2000.0 (price at 2,500: 1.0; S per question 0.0) -> False
  - metaqa/SRK: reach 0.9582, price 0.6979, path retention 0.9741 (8105 golds), kept links per question 6569.2 of 9413.0 (price at 2,500: 0.26559; S per question 3642.1) -> False
  - webqsp/SRK: reach 0.9372, price 0.975, path retention 0.9632 (6632 golds), kept links per question 2186082.3 of 2242206.9 (price at 2,500: 0.00111; S per question 36939.0) -> False
  - hotpotqa/RK: reach 0.9635, price 0.1486, path retention 0.9428 (594 golds), kept links per question 8647.8 of 58203.9 (price at 2,500: 0.04295; S per question 0.0) -> True
  - hotpotqa/SRK: reach 0.9894, price 0.2572, path retention 0.9849 (595 golds), kept links per question 6372.8 of 24775.2 (price at 2,500: 0.10091; S per question 2945.5) -> True
  - 2wiki/RK: reach 0.9714, price 0.3556, path retention 0.9321 (1120 golds), kept links per question 64376.4 of 181014.9 (price at 2,500: 0.01381; S per question 0.0) -> True
  - 2wiki/SRK: reach 0.98, price 0.3835, path retention 0.9625 (1120 golds), kept links per question 58910.2 of 153619.6 (price at 2,500: 0.01627; S per question 6344.4) -> True
- rule_P5 (phase 5 best; pr+vo+ur): NOT_UNIVERSAL (4/9 cells SCREEN_IN)
  - multihoprag/RK: reach 0.9995, price 0.9358, path retention 1.0 (121 golds), kept links per question 520.1 of 555.7 (price at 2,500: 1.0; S per question 0.0) -> False
  - musique/RK: reach 0.953, price 0.9089, path retention 0.9455 (1469 golds), kept links per question 29628.6 of 32599.3 (price at 2,500: 0.07669; S per question 0.0) -> False
  - squad/RK: reach 0.9201, price 0.8226, path retention 0.9481 (77 golds), kept links per question 1645.1 of 2000.0 (price at 2,500: 1.0; S per question 0.0) -> False
  - metaqa/SRK: reach 0.9552, price 0.6858, path retention 0.9685 (8105 golds), kept links per question 6455.4 of 9413.0 (price at 2,500: 0.26559; S per question 3642.1) -> False
  - webqsp/SRK: reach 0.9371, price 0.9745, path retention 0.9632 (6632 golds), kept links per question 2184920.8 of 2242206.9 (price at 2,500: 0.00111; S per question 36939.0) -> False
  - hotpotqa/RK: reach 0.9677, price 0.1474, path retention 0.9478 (594 golds), kept links per question 8581.2 of 58203.9 (price at 2,500: 0.04295; S per question 0.0) -> True
  - hotpotqa/SRK: reach 0.9911, price 0.2561, path retention 0.9882 (595 golds), kept links per question 6345.8 of 24775.2 (price at 2,500: 0.10091; S per question 2945.5) -> True
  - 2wiki/RK: reach 0.9713, price 0.355, path retention 0.9429 (1120 golds), kept links per question 64265.1 of 181014.9 (price at 2,500: 0.01381; S per question 0.0) -> True
  - 2wiki/SRK: reach 0.9802, price 0.3829, path retention 0.9679 (1120 golds), kept links per question 58819.7 of 153619.6 (price at 2,500: 0.01627; S per question 6344.4) -> True
- rule_C2 (C2 (per-question caps); K1): NOT_UNIVERSAL (3/9 cells SCREEN_IN)
  - multihoprag/RK: reach 1.0, price 1.0, path retention 1.0 (121 golds), kept links per question 555.7 of 555.7 (price at 2,500: 1.0; S per question 0.0) -> False
  - musique/RK: reach 0.9932, price 0.9201, path retention 0.9598 (1469 golds), kept links per question 29995.5 of 32599.3 (price at 2,500: 0.07669; S per question 0.0) -> False
  - squad/RK: reach 0.9827, price 0.8878, path retention 0.961 (77 golds), kept links per question 1775.7 of 2000.0 (price at 2,500: 1.0; S per question 0.0) -> False
  - metaqa/SRK: reach 0.9781, price 0.8433, path retention 0.9993 (8105 golds), kept links per question 7938.4 of 9413.0 (price at 2,500: 0.26559; S per question 3642.1) -> False
  - webqsp/SRK: reach 0.9438, price 0.0511, path retention 0.97 (6632 golds), kept links per question 114505.2 of 2242206.9 (price at 2,500: 0.00111; S per question 36939.0) -> True
  - hotpotqa/RK: reach 0.9932, price 0.5483, path retention 0.9933 (594 golds), kept links per question 31911.5 of 58203.9 (price at 2,500: 0.04295; S per question 0.0) -> False
  - hotpotqa/SRK: reach 0.9984, price 0.6052, path retention 0.9983 (595 golds), kept links per question 14993.8 of 24775.2 (price at 2,500: 0.10091; S per question 2945.5) -> False
  - 2wiki/RK: reach 0.9769, price 0.2566, path retention 0.9759 (1120 golds), kept links per question 46441.4 of 181014.9 (price at 2,500: 0.01381; S per question 0.0) -> True
  - 2wiki/SRK: reach 0.9838, price 0.2694, path retention 0.992 (1120 golds), kept links per question 41379.9 of 153619.6 (price at 2,500: 0.01627; S per question 6344.4) -> True
- rule_C2r (C2r (per-question relative caps); K3): NOT_UNIVERSAL (0/9 cells SCREEN_IN)
  - multihoprag/RK: reach 0.9993, price 0.9, path retention 1.0 (121 golds), kept links per question 500.2 of 555.7 (price at 2,500: 1.0; S per question 0.0) -> False
  - musique/RK: reach 0.9659, price 0.8718, path retention 0.9905 (1469 golds), kept links per question 28418.6 of 32599.3 (price at 2,500: 0.07669; S per question 0.0) -> False
  - squad/RK: reach 0.9645, price 0.8547, path retention 0.961 (77 golds), kept links per question 1709.5 of 2000.0 (price at 2,500: 1.0; S per question 0.0) -> False
  - metaqa/SRK: reach 0.9537, price 0.876, path retention 0.9888 (8105 golds), kept links per question 8245.4 of 9413.0 (price at 2,500: 0.26559; S per question 3642.1) -> False
  - webqsp/SRK: reach 0.9732, price 0.9004, path retention 0.9929 (6632 golds), kept links per question 2018876.5 of 2242206.9 (price at 2,500: 0.00111; S per question 36939.0) -> False
  - hotpotqa/RK: reach 0.9825, price 0.8829, path retention 0.9933 (594 golds), kept links per question 51390.5 of 58203.9 (price at 2,500: 0.04295; S per question 0.0) -> False
  - hotpotqa/SRK: reach 0.9952, price 0.8994, path retention 0.9966 (595 golds), kept links per question 22282.3 of 24775.2 (price at 2,500: 0.10091; S per question 2945.5) -> False
  - 2wiki/RK: reach 0.9797, price 0.8734, path retention 0.9955 (1120 golds), kept links per question 158106.5 of 181014.9 (price at 2,500: 0.01381; S per question 0.0) -> False
  - 2wiki/SRK: reach 0.9865, price 0.8725, path retention 0.9991 (1120 golds), kept links per question 134034.6 of 153619.6 (price at 2,500: 0.01627; S per question 6344.4) -> False