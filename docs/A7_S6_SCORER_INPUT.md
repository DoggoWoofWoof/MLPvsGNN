# A7: S6's scorer, as a fixed input to both models

Declared 17 October 2026, before any number. The seventh reopening of docs/VERDICT_AUDIT_2026_10_17.md (section 2,
row 7), under docs/SCREENS.md section 3b. Code: `scripts/a7_s6_cols.py` (the scorer and its columns) and
`outputs/mp_unified/za7s6.py` (the arms, the screen and the call). Not run yet.

## 1. Why this, now

S6 (chainscore17 to chainscore31, 4 to 7 October) learned a scorer over a question's typed chains: which relation
paths out of the seeds the question follows, at several granularities of the relation vocabulary (the transforms).
- **It transferred.** S6's standalone scorer gained 6.43 points on webqsp zero-shot (significant), the largest
  zero-shot gain any label-free rule gave on a KB graph (docs/VERDICT_AUDIT_2026_10_17.md, row 7).
- **It never reached the models.** T1 (17 October, NOT_ADOPTED) tried only S6's transforms as a metaqa training
  augmentation of the lean models (zrckp 2 LOSS on webqsp zero-shot, zspk 0 and 0). The scorer's output was never an
  input. Under section 3b the idea stays open until it reaches both models.

**The form chosen: fg-gr, not em-gr.** chainscore20's K5 (outputs/mp_unified/lean/cs20.json) compared the EM-coupled
scorer em-gr with its T = 0 control fg-gr:

| read | em-gr − fg-gr, R@5 [95%] | call |
| --- | ---: | --- |
| metaqa (in domain) | +0.0112 [+0.0064, +0.0156] | HELPS |
| webqsp (zero-shot) | −0.0208 [−0.0520, +0.0094] | SAME |
| webqsp_sf (zero-shot) | −0.0189 [−0.0327, −0.0043] | HURTS |

- **The EM rounds help in domain and hurt zero-shot.** The input is meant to carry transfer.
- **em-gr also breaks the MLP rule.** Its rounds reweight chains by a softmax of s over the question's nodes, a
  score normalised over the question's rows. fg-gr has no round: s(v) is v's own.
- fg-gr's own reads in chainscore20, R@5: metaqa 0.638, webqsp 0.257, webqsp_sf 0.286.

## 2. The form

**The scorer** is chainscore19's joint model at T = 0 (fg-gr) with chainscore21's length offset (lb):

    s(v) = f(v) + γ log(ε + Σ_c p0(c) m_c(v)),    p0 = softmax over the row's chains of g(c) − log C_level,length

- f is an MLP (13 → 64 → 64 → 1) over v's node features: question cosine, cosine to the seeds' mean, first-stage
  rank, seed distances, degree, walk masses, and the best neighbour's question cosine.
- g is an MLP over a chain's pair features. The chains are the walks out of the bucket-0 seeds at every level of the
  relation hierarchy; m_c(v) is chain c's walk mass at v.
- γ = softplus(γ_r) and ε = exp(leps) are learned scalars.

**The builds** are chainscore19's, of P1c2's looks (outputs/p1c2/look), with S6's settings:

| graph | hops | levels | on the fit carve also |
| --- | ---: | --- | --- |
| metaqa | 3 | 2, 4 | al4, sp4 (rows even / odd, hops 2, levels 2,4,16, regimes 1,0.05,0); mg3 (hops 3, level 2, six regimes); rf (rows even, hops 3, levels 2,4, regimes 1,0.05,0) |
| webqsp | 2 | 4, 16, 64, 256 | (webqsp never trains) |
| passages | 2 | 4, 16, 64 | |

- **Passages are pseudo-typed** by chainscore21's typing: 256 clusters (spherical k-means) of the endpoint
  projections of the carve's structural triples.
- **Regimes:** 1, 0.25, 0.05, 0.01, 0.002, 0 on a fit carve; 1 (the whole carve) on s1eval.
- **One row per question,** in the look's order, with the look's whole pool. The build refuses otherwise.

**The fits:** one scorer per screen split and fold (four in all).
- **Training data:** the split's four training datasets' fit builds (metaqa's five), never its held-out dataset.
- **Folds:** fold f holds the questions whose sha256('dataset:question id') is ≡ f mod 2. Each fit trains on its
  fold less 20% held out (P5b2's split_of, seed 0).
- **Each example:** the dataset uniform among the four (chainscore21's draw), then chainscore19's draw within it.
- **Optimisation:** Adam 1e-3, weight decay 1e-4, 12 epochs of 5,960 examples in batches of 64, one thread,
  deterministic. Inputs are standardised by metaqa's id fit build.
- **The kept epoch** is the one (1 to 12) whose dataset-balanced held-out objective (chainscore19's node loss plus
  chain loss, regime 1) is lowest; ties go to the earlier.

**The columns:** four per cache row of P1c2's cache, for each split that reads the carve. Each question's columns come
from the scorer of the other fold (it never trained on that question):

| column | value |
| --- | --- |
| f | f(v) |
| lm | log(ε + Σ_c p0(c) m_c(v)) |
| has | 1 when Σ_c p0(c) m_c(v) > 0 |
| seed | 1 on the question's bucket-0 seeds |

They are stored as float16. An s1eval carve is built inside its item and its build removed.

**Into both models: the pointwise head.** Linear(4, 16), GELU, Linear(16, 1): 97 parameters, the last layer zero at
the start, the first drawn from its own generator (seed + 7701). From the same seed each arm's trunk is its base's and
each arm starts as its base bit for bit.

**The MLP rule.** The columns are the row's own and fixed before either model scores a row:
- p0 is a softmax over the question's chains, from chain features, not over its rows;
- no column reads a model's score or another row's score;
- m_c(v) is a fixed walk of the pool, computed with the pool.

The MLP reads them as fixed numbers; it passes no message.

## 3. The arms

| arm | model | carve |
| --- | --- | --- |
| zrckps | zrckp's model plus the head at the top (MRO: ZRCKPS, ZRCKP, …) | zrckp's (`zp1c3.KTChainCarveZRC`) with the columns beside |
| zspks | zspk's model with the head added to the KT head before zsp's propagation reads the score (MRO: ZSPKS, ZProp, A7Mix, KTMix, ZRM, …) | zspk's (`zp1c3.KTLinkCarveBase`) with the columns beside |

Each split's fits read that split's columns only (`outputs/a7/cols/<split>/…`). A fit of L-musique never sees a
scorer that trained on musique, nor one of L-hotpotqa on hotpotqa.

## 4. The fits and the call

- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-a7-<arm>`, `-hp`), on P1c2's roots with
  P1c3's KT input. Card fit settings: variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128.
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split, which decides:
  - zrckp: scr-p1c3p-zrckp and -hp;
  - zspk: scr-p1c3-zspk and -hp.

  Step 1's fit is reported beside. The re-call's base screens are L1's.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), per arm. GAIN,
  LOSS and WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A7's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- webqsp and metaqa, both splits and arms (S6's reads);
- musique in domain (L-hotpotqa), both arms: Claim 1's read;
- hit@1 and FC@5 per read;
- per fit: the kept epoch, the held-out objective per dataset, γ and ε;
- per column part: the quantiles of f and lm, the shares of has and seed, the float16 error.

**Serving cost** (reported later, not part of this call): the columns need the pool's chains and the scorer's two MLPs
at query time; the timed form is declared with the full run if one follows.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools (the scorer and its columns built again on P1g-b's looks), then its full run |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A7k, the back-off form: the four columns split by the graph's kind (the KB graphs' and the pseudo-typed passage graphs'; eight columns, the other kind's four at 0), so the KB graphs' chain weights cannot move the passage graphs' head weights; both models |
| NOT_ADOPTED, with no GAIN | A7e: on zspk, chainscore20's carried scorer ena-gr (its GNN f with the EM rounds; K1 above em-gr) in the scorer's place, a message-passing form the GNN may read; on zrckp, the fg-gr scorer fitted on metaqa's five builds alone (S6's filed setting) |
| INCOMPLETE | the missing re-call is run; nothing else changes |

- **If both arms pass,** the columns join both lean models.
- **If only one passes,** its full run is declared; the next file decides whether F1's rule needs the other model to
  read the columns too.
- **What does not happen.** No model change is adopted on today's pools (the user's rule of 13 October).

## 6. How it runs

**Self-tests, both passed before this commit.**

`python scripts/a7_s6_cols.py --selftest` (367 s) checks, on chainpop17's typed toy KB as metaqa and chainscore21's
untyped toy as 2wiki, each with a fit and an s1eval carve and two step-1 cache parts:
- metaqa's five builds and the typed passage build hold the look's rows (even / odd for al4, sp4 and rf), with A7's
  records; chainscore21's temporary typed root is removed;
- each fold's scorer trains on its questions less the held-out fifth (counts checked), keeps the epoch with the
  lowest held-out objective, and repeats bit for bit;
- each question's columns come from the other fold's scorer. They match a float64 brute force from that scorer's
  weights (f, the softmax over chains, the mass sum, the flags) on every row of all four carves, and differ from the
  own fold's on most rows;
- s from the parts is chainscore19's forward bit for bit; the seed column is the look's bucket-0 seeds;
- an s1eval build is made inside the item and removed;
- the refusals: a cache part from another look, a part whose n is not its record's, a build of another look, a fit
  under other settings, a carve no split reads, a transform off metaqa's fit carve, a split whose held-out dataset
  would train the scorer.

`python outputs/mp_unified/za7s6.py --selftest` (10 s) checks:
- the arms' MROs; the base arms unchanged; 97 head parameters; the builder's constants and settings are the model
  module's;
- a carve reads only under `on_split`, only its split's columns, only columns built on its own step-1 parts under the
  builder's settings;
- a batch's columns are its rows' own;
- each arm starts as its base bit for bit; with the head set, zrckps adds it and zspks's propagation reads the score
  with it added; without the columns each is its base;
- toy fits through lean_gpu's loop repeat bit for bit and move the head; without the columns each fit is its base's;
- the refusals; relz's pair and re-call per arm, stamped A7's; the verdict's outcomes and next forms.

**Host probes** (one thread, regime 1, P1c2's looks; 17 October):

| dataset | rows | time | npz | peak |
| --- | ---: | ---: | ---: | ---: |
| hotpotqa | 256 | 19 s | 3.7 MB | 0.15 GB |
| squad | 479 | 33 s | 12.7 MB | 0.13 GB |
| 2wiki | 227 | 19 s | 3.7 MB | 0.11 GB |
| musique | 22 | 44 s | 3.7 MB | 0.69 GB |
| webqsp | 33 | 53 s | 15.5 MB | 0.41 GB |
| metaqa | 33 | 2 s | 3.8 MB | – |

The whole chain is a few CPU-hours; musique's s1fit build and webqsp's s1eval build are the longest (about 45 and
40 minutes).

**Host items** (`a7-*`), after A6's verdict:
- 9 builds, 1 CPU each: metaqa's fit carve five times (id and the four transforms), squad, hotpotqa and 2wiki fit,
  musique s1fit;
- 4 scorer fits (two splits, two folds), 1 CPU each, each after its split's builds;
- 11 column items, 1 CPU each: the five fit carves (on their stored builds) and the six s1eval carves (built in the
  item), each after the four fits;
- the 17 screen items, the trains gated on their split's fit-carve columns and the reads on the six s1eval carves'.
  GPU sizing as A3's to A6's, from the base arms' measured peaks: L-musique trains `--frac 0.20` (share 0.22), -hp
  trains and the reads `--frac 0.27` (0.29); memory: trains 4.5 GB (zspks-hp 7 GB), reads 6 GB.
