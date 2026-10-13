# P2c part 2: our LLM-free triples as a fourth link family (a screen, closed NO_GAIN)

Filed 13 October 2026, about 14:50. Part 2 of P2c ([P2C_LLM_FREE_TRIPLES.md](P2C_LLM_FREE_TRIPLES.md), section 9),
read under P1e's chosen rule, KIND_T ([P1E_ANCHOR_KIND.md](P1E_ANCHOR_KIND.md), section 8). This is a screen on the
fit carves, not a declared stage. It read no s1sel or s1eval question, and no model trained or read. Under the
fast-iteration rule ([SCREENS.md](SCREENS.md)), a full run follows only a promising screen. This screen is not
promising, so no stage is declared and nothing is adopted.

## 1. Question

Part 1 read our triples graph, built without an LLM, on B1's musique and 2wiki passages. In the pool walk it added
more than every LLM-built graph: +8.4 R@5 on musique against +3.7 to +4.8, and +3.8 on 2wiki against +1.5.

Part 2 asks the same on our own corpora, in full. Does the P form of our parse, as a fourth link family beside
structural, ner and knn, put more golds in fewer nodes under the pool's rule?

## 2. The fourth family

- **The parse** is part 1's version 1, run on the host over every passage of the four passage corpora (items
  fee0c90). The knowledge bases have no passages to parse.
- **One vocabulary.** Each part's phrases are keyed by the first 8 bytes of blake2b over their UTF-8 bytes, so a
  phrase gets the same key in every part.
- **The P form** is part 1's: two passages are linked when they share a phrase that 2 to 25 passages hold. A pair is
  linked once, undirected, with no self-loop. There is no new constant.

| corpus | passages | phrases | phrases with df 2-25 | edges | edges per passage | build |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| squad | 20,233 | 285,243 | 48,944 | 628,568 | 31.1 | 1 s (laptop) |
| musique | 117,534 | 976,618 | 178,594 | 2,044,439 | 17.4 | 6 s (laptop) |
| hotpotqa | 5,233,329 | 16,993,497 | 3,375,233 | 51,043,402 | 9.8 | 228 s (host) |
| 2wiki | 5,989,847 | 26,554,732 | 5,600,517 | 84,848,015 | 14.2 | 481 s (host) |

**Most of its edges are new.** On squad and musique, 74% and 66% of its pairs are in none of the three families.
It holds 83% and 77% of the ner family's pairs, since a shared name is also a shared phrase.

| corpus | its pairs also in structural | in ner | in knn | in any of the three | of ner's pairs, in it |
| --- | ---: | ---: | ---: | ---: | ---: |
| squad | 0.065 | 0.207 | 0.036 | 0.258 | 0.834 |
| musique | 0.058 | 0.300 | 0.044 | 0.342 | 0.766 |

## 3. The screen

**The walks** are step 4e's kernel over four families. A node's mass goes 1/nf to each of its nf non-empty families
and, within a family, equally to its neighbours or by P1e's steer. With an empty fourth family the walk is P1b's walk
and P1e's steered walk, bit for bit. The prototype checked this on 64 of musique's s1sel questions, reading no gold:
both walks, plain and steered, differed on none.

**The rule with the fourth family** takes every walk on the 'all' union (A1all, Kall, A1Tall and KTall) on the four
families. Everything else is P1e's.

**The questions** are the first 600 of each corpus's fit carve (musique's s1fit), the models' training questions. P1a's
bar is recomputed there. A point is one of 15 sizes N and two measures, and it counts when the rule is more than
0.002 below the bar.

**Points below the bar** (the largest gap in brackets), then ALL at N = 25, N = 100 and |I_q|. |3 is the three
families and |4 adds the fourth:

| order | squad | musique | hotpotqa | 2wiki | points |
| --- | --- | --- | --- | --- | ---: |
| KIND_T\|3 (the pool) | 2 (-0.005): 0.985, 0.997, 0.993 | 0: 0.728, 0.880, 0.983 | 2 (-0.013): 0.872, 0.957, 0.962 | 0: 0.862, 0.942, 0.943 | 4 |
| KIND_T\|4 | 2 (-0.007): 0.983, 0.997, 0.993 | 1 (-0.003): 0.730, 0.880, 0.982 | 2 (-0.015): 0.873, 0.955, 0.962 | 0: 0.855, 0.945, 0.943 | 5 |

The fourth family moves the pool's ALL by -0.007 to +0.002 at N = 25, by -0.002 to +0.003 at N = 100, and by at most
0.001 at |I_q|. It adds one point below the bar. On 600 questions these moves are within the noise: the standard
error of a share near 0.86 is about 0.014.

**The walks it changes,** beside (ALL at N = 25, N = 100 and |I_q|):

| walk | squad | musique | hotpotqa | 2wiki |
| --- | --- | --- | --- | --- |
| A1all\|3 | 0.962, 0.995, 0.985 | 0.562, 0.813, 0.972 | 0.745, 0.938, 0.932 | 0.463, 0.815, 0.815 |
| A1all\|4 | 0.965, 0.997, 0.988 | 0.582, 0.813, 0.972 | 0.748, 0.933, 0.927 | 0.467, 0.758, 0.765 |
| A1Tall\|3 | 0.967, 0.995, 0.987 | 0.602, 0.825, 0.972 | 0.775, 0.928, 0.925 | 0.462, 0.745, 0.748 |
| A1Tall\|4 | 0.980, 0.995, 0.992 | 0.617, 0.822, 0.972 | 0.765, 0.923, 0.922 | 0.455, 0.713, 0.722 |
| Kall\|3 | no links | 0.265, 0.382, 0.710 | 0.617, 0.733, 0.727 | 0.875, 0.948, 0.947 |
| Kall\|4 | no links | 0.265, 0.372, 0.670 | 0.605, 0.728, 0.717 | 0.848, 0.948, 0.948 |
| KTall\|3 | no links | 0.363, 0.535, 0.820 | 0.697, 0.790, 0.783 | 0.878, 0.945, 0.945 |
| KTall\|4 | no links | 0.357, 0.495, 0.793 | 0.695, 0.772, 0.770 | 0.842, 0.943, 0.943 |

- The walks from the first stage's seeds (A1) gain at the head on the two small corpora: +0.003 to +0.020 at N = 25
  on squad and musique.
- On 2wiki they lose from N = 100 (A1all 0.815 to 0.758, A1Tall 0.745 to 0.713).
- The walks from the links (K) lose on all three corpora with links. The fourth family takes a quarter of each
  node's mass, and on 2wiki and hotpotqa it holds more edges than the other three together.
- Under the pool's rule, the gains and losses cancel.

**Cost** (the hotpotqa and 2wiki screens, measured by rx): 5.4 GB and 9.6 GB at peak, 1,000 and 1,483 CPU seconds.

## 4. What it decides

- **NO_GAIN: the fourth family is not adopted, and no full run follows.** The pool stays P1e's KIND_T on the three
  families, and the links stage closes with them.
- **Part 1's gain does not carry to our corpora.** On B1's small corpora, our triples added more in the pool walk
  than every LLM graph. On our full corpora, with three families and a rule that fuses the first stage's lists with
  the walks, they add no golds per node. Most of their edges are new, but those edges do not move golds into the
  pool's head.
- **Version 2 of the parse** (part 1, section 9) is not pursued for the pool. Its aim was 2wiki's phrases in
  HippoRAG 2's search, not our pool.

## 5. Records

- The screen's curves: `outputs/p2c/dev/p2c2_explore_out/<corpus>.json` (fit carves only, a few KB each).
- The scripts: `outputs/p2c/dev/p2c2_explore.py` and `p2c2_proto.py` (83752e2). They import P1e's script unchanged.
- The edge overlap was computed on the laptop from the same build and the family stores
  (`outputs/m3b/csr/<corpus>_<family>.npz`).
- The parse parts hold passage text and are not committed; their records (counts) are.
