# U2p: the learned link score as the walks' edge weight

Declared 25 October 2026, after U2m (a)'s verdict and before any code or number. It is the form U2o's filing names
([U2O_K1_I_ORDER.md](U2O_K1_I_ORDER.md), section 7). Code: `outputs/mp_unified/zu2p.py`, which imports zu2m.py,
zu2m_b.py, zu2m_a.py and zu2m_a_types.py (the carried relational walk and its EM types), zu2o.py (the pinned scorers
and K1's filed columns) and zu2k.py (the scores) unchanged; its self-test also reads zu2m_d.py's toy graphs. Outputs
go only under `outputs/u2p/`. Pools only: nothing
trains here. The scorers are U2k's filed ones, unchanged, and no weight reads a gold or the target dataset's
hyperlinks.

## 1. The question

U2o found that K1, as the I's row order, adds no gold to the carried pool: KIND_T's walks already reach the golds the
I gains. What the pool misses on the datasets that set its level is mostly BUDGET: golds the walks reach but rank
below the budget (hotpotqa 38 of 69 misses, 2wiki 31 of 50; U2o, section 7).

The carried walk is now U2m (a)'s pick (U2M_POOLS_THREE_FAMILIES.md, section 18): `own/kind_t/kt_cos:ada_f12_k0@0.85`
at G = 0 with RWe at β = 0.5, level 0.955 for 1,809.6 rows. It spreads a node's mass over its W2_S links in proportion
to ω_q of each link's EM type. Nothing in it reads whether a link is a hyperlink. U2k's score does, zero-shot, at AUC
0.825 (hotpotqa) and 0.832 (2wiki). U2p asks whether also weighting each link by that score ranks the reached golds
higher, and so raises the pools or makes them cheaper.

## 2. The weight

- **The score.** On a passage graph's structural family (W2_S, the file U2m swaps in), an entry u–v carries x(u, v),
  the linear score (the logit) of s(u, v), the higher of the pair's directed W2 link scores (U2k, section 4). The
  scorer is the zero-shot one, as U2k and U2o read it: A (fitted on 2wiki) on musique, squad and hotpotqa; H (fitted
  on hotpotqa) on 2wiki. A sibling-only entry has no score.
- **The score types.** z(u, v) is x standardized over the graph's scored entries (minus their mean, over their
  standard deviation, unweighted). The scored entries are cut into 64 types at the quantiles of z (equal counts; an
  entry on a boundary goes to the lower type). Each type's value z̄_b is the mean z of its entries. Sibling-only
  entries take a null type.
- **The score weight.** ω(b) = exp(β × z̄_b), and 1 on the null type. It is the same for every question.
- **The walk.** Within the structural family, u's pushed mass goes to v in proportion to

      ω_a,q(t(u→v)) × ω(b(u–v)) × s(v)

  - ω_a,q is the carried walk's weight: RWe at β = 0.5, with (a)'s filed EM types and centres, unchanged (section 14
    of the U2m file). s(v) is the walk's own weight, 1 or P1e's steer.
  - The kernel is (a)'s relational kernel, unchanged. Its mode 1 reads one weight per entry's type. Here an entry's
    type is the pair (its EM type, its score type), 65 × 65 of them, and the weight over the pairs is the product
    ω_a,q(t) × ω(b).
  - The ner and knn families keep ω = 1.
- **Which walks.** The five the carried recipe reads (A1st, A1all, Kst, KTst and KTall), with their restarts, α, ε,
  NMAX and the equal split over a node's non-empty families.
- **The KBs** (metaqa, webqsp) have no score. Their candidates are (a)'s filed ones at RWe, β = 0.5, at every β, and
  they are not walked again.
- **β = 0** gives ω(b) = 1 on every edge. Multiplying by 1 is exact, so the walk is (a)'s pick's, bit for bit.

## 3. The candidates and the choice

- **The grid.** β is 0.5, 1 or 2: three universal points, plus β = 0. 64 types is fixed; another count, or a weight
  read without types, is a later form.
- **The candidates** are (a)'s pick's family: the 8 points of the carried frontier at G = 0, on RWe at β = 0.5, each
  crossed with β (named `<point>|G0|RWe@0.5|beta0` and `…|LW@<β>`). The order is β = 0 first, then β ascending,
  then the points' order, so ties go to the carried walk and then to the weaker weight. Each candidate's order is the
  carried order rebuilt on the weighted walks; KIND_T's switch, RRF5T's R0, N1, the cosine order of U_0, U2e's budgets
  and P1h's cut are unchanged.
- **The choice** is U2m's section 7 rule, with U2e's cost (P1f's weights), the levels, the cap of 1,953.5 and (c)
  and (e)'s filed ceilings.

| verdict | when |
| --- | --- |
| LINKWEIGHT_RAISED | the pick's level is above 0.955 |
| LINKWEIGHT_CHEAPER | the level is 0.955, β > 0, and the cost is below 1,809.6 |
| NO_GAIN | otherwise: (a)'s pick stays |

**Carried forward.**
- RAISED or CHEAPER: the weighted walk is the pools' walk, and U2m's (d) arrays are computed on it for U2n, in their
  own file.
- NO_GAIN: (a)'s walk stays.
- Either way, s(u, v) remains an input for both models (U2k, section 8), and the score types are a candidate edge type
  for U2n's GNN.
- E1 (U2m, section 19) and U2p change different parts of the walk. If both gain, their combination is declared in its
  own file before it runs.

## 4. Checks

Any failure stops the dataset, and nothing is written.
- **The scores.** zu2p.py rebuilds x(u, v) on every W2_S entry with U2k's code (`zu2k.pair_scores`), and the
  standardization of the features must equal U2k's graded one. On musique, hotpotqa and 2wiki, sorting each row by
  x (sibling-only entries last, ties to node id) must give U2o's filed K1 columns exactly, so the weights read the
  scores U2o filed. squad has no K1 file; there the standardization check alone ties the scores to U2k's.
- **The types.** Every scored entry has one type in 0..63 and every sibling-only entry the null type. The types'
  counts, cuts and z̄ are filed with the array's sha256. s(u, v) is the higher of the pair's two directed scores, so
  u–v and v–u share one score type (the self-test checks it on toy graphs). The EM types are (a)'s filed ones: their
  record's sha256 equals the one (a)'s stats read.
- **β = 0.** For every question, the β = 0 candidates' rows and held golds equal (a)'s filed table at RWe, β = 0.5.
  In the verdict, the β = 0 candidates equal (a)'s filed values on all six datasets.
- **The kernel.** With ω(b) = 1, the paired weights equal (a)'s, so the kernel's p and its top-NMAX order equal
  (a)'s kernel at RWe bit for bit (self-test on toy graphs with 65 × 65 pair types). With ω(b) ≠ 1, its p equals a
  plain Python push within 1e-12 relative.
- **No gold before the choice.** The scores, types, weights and walks take no gold argument.
- **Self-test** (`--selftest`), before the commit that adds the code: the binning against a brute force on toy scores
  (ties and boundaries included), the symmetry, the null type, the pairing, and the verdicts.
- **Development runs** (`--limit`) write nothing and log no number.

## 5. Beside, deciding nothing

- Per dataset and β: ALL and rows at the carried pick's point.
- The pick's misses by U2m's causes (TAU_CUT, BUDGET, UNREACHED), against the carried pick's, with the recovered and
  lost golds by cause.
- Per dataset and β: the share of questions whose KIND_T order the weight changes.
- Per dataset: the types' z̄ from the lowest to the highest type, and the share of entries in the null type.
- webqsp's s1eval, unchanged, as everywhere.

## 6. Where it runs

- **`types D`** builds the scores and types on the laptop, for musique, squad, hotpotqa and 2wiki (U2o's graph stage
  built the same scores: 29 s on musique, 574 s on hotpotqa, 1,370 s on 2wiki). It writes
  `outputs/u2p/types/<D>.npz` and `.json` (the array stays uncommitted).
- **`stats D`** walks the four passage datasets at β in 0.5, 1 and 2, and at β = 0 for the check. It walks four
  configurations where (a) walked three on these graphs, so it costs about 4/3 of (a)'s stats (1,853 s on squad,
  3,399 s on musique, 3,337 s on hotpotqa and 3,882 s on 2wiki, the last two in 4 shards). It writes
  `outputs/u2p/stats/<D>__<carve>.npz` and `.json` (the arrays stay uncommitted); `--shard I/K` and
  `merge D --parts K` split a dataset into question ranges and join them into the file one run writes.
- The laptop runs them, or the lab host once it is reachable. W5's credit is too little for Modal. A machine cannot
  change a number: at β = 0 every question must reproduce (a)'s filed rows and held golds, or the run stops.
- As with U2o's code, `types musique` and a 30-question development run on musique run before the code's commit.
  The types hold no gold and no pool number, and the development run logs none.
- **`verdict`** runs on the laptop and writes `outputs/u2p/verdict.json` and `.md`.

The four passage corpora are all Wikipedia, so the score's transfer is between Wikipedia corpora only (U2k's caveat).

A verdict closes only the form tested (docs/SCREENS.md, section 3b): here U2k's score in 64 types as a question-free
factor on (a)'s walk, with these β. A question-dependent weight that reads the score, a weight read without types,
and a prune whose cut each graph reads are different forms, and none is closed.
