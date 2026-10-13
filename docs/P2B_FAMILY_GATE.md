# P2b: one link rule for all six datasets, the families' walks weighted per question (a diagnostic)

Declared 13 October 2026, about 07:40, before any number of the stage. P2's second stage in
[PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md), after [P2A_LINK_FAMILIES.md](P2A_LINK_FAMILIES.md). It reads the pools
only: no model trains or reads here, and nothing is adopted until the call below says so.

## 1. What P2a left

Each dataset's pools walk a graph picked per dataset (its regime): the structural family alone on metaqa, hotpotqa
and 2wiki, all three families on musique and webqsp, and no walk on squad. That pick is a per-dataset setting, and
P2a part 1 shows it is not even the best one everywhere. Under A3 (today's restart), ALL at each question's size of
today's pool (|I_q|), s1eval:

| dataset | no walk | structural alone | all three | the regime |
| --- | ---: | ---: | ---: | --- |
| metaqa | 0.261 | 0.800 | 0.730 | structural |
| squad | 0.981 | 0.955 | 0.964 | no walk |
| musique | 0.760 | 0.837 | 0.884 | all three |
| hotpotqa | 0.691 | 0.857 | 0.799 | structural |
| 2wiki | 0.462 | 0.772 | 0.623 | structural |
| webqsp | 0.412 | 0.829 | 0.759 | all three |

P2a's results proposed a rule that walks the structural family first and ner and knn after it. Its own table rules
out any fixed order of the families:
- On musique, ner alone (0.855) is above the structural family alone (0.837), and all three together are best
  (0.884). A structural-first order costs musique its head.
- On webqsp, the structural family alone is 0.070 above the regime's three families.
- On squad, every walk is below no walk (0.955 to 0.964 against 0.981).

No single weighting of the families serves musique and 2wiki together. On 2wiki, adding ner or knn to the structural
walk costs 0.11 at |I_q|. On musique, leaving either out costs 0.008 to 0.032.

So the universal form is a choice the question's own features make: walk the families in both ways, and let one gate
weight the walks per question. P1b's mixture already does this for restarts, with weights set by label-free features
and fit on the five training datasets. P2b gives that gate the families' walks as its sources, on all six datasets,
and drops the per-dataset regime from the pools.

## 2. The construction (MXU)

- **Nine sources per question, on every dataset.** Each is a distribution over nodes. The walks use step 4e's kernel
  (its alpha and eps), with nothing excluded, over M3B's family stores (the ones the pools read):

  | zero-step (P1b's) | walked on the structural family alone | walked on all three families |
  | --- | --- | --- |
  | S0: today's seeds, uniform | S_st | S_all |
  | L0: our exact-match linked nodes, uniform | L_st | L_all |
  | R0: the RRF-weighted dense and SPLADE top-1,000 lists | R_st | R_all |

  - A walk's masses are its top 7,000 nodes', scaled to sum to 1, as in P1b.
  - The regime is never read. squad's families are walked as every other dataset's are.
  - On every dataset, P1b's own six sources are three of the zero-step ones plus one column of walks: S_st, L_st and
    R_st where the regime is the structural family, S_all, L_all and R_all where it is all three, and the zero-step
    ones again on squad.
- **The mixture.** m_q = sum over k of w_k(q) m_k, with w(q) = softmax(W x(q)). x(q) holds twenty-four label-free
  features:
  - P1b's nine features of the lists and links: the number of seeds and of linked nodes (log); whether there is no
    linked node; the dense and SPLADE top scores and their gaps to the tenth; the share of linked nodes inside the
    lists' top 100; the share of seeds that are linked;
  - each of the six walks' top mass (log) and top-10 mass;
  - for each family, the mean over the seeds of log(1 + the seed's degree in that family). This is how dense each
    family is where the question starts, read from the corpus's own graph.

  No feature reads a gold or names a dataset.
- **The pool at N** is the mixture's top N, ties by node position.
- **One parameter set for all six:** W, nine sources by twenty-five weights (the features and a bias).

## 3. The fit: P1b's EM, unchanged

- **Data:** the fit carves of the five training datasets, whole. That is fit on metaqa, squad, hotpotqa and 2wiki,
  and s1fit on musique. s1sel and s1eval are never fit. webqsp never fits, so its read is zero-shot.
- **The likelihood** is that of each question's golds under its mixture. Every question weighs the same within its
  dataset, and every dataset weighs the same. A question's golds share its weight, and a gold that no source reaches
  carries none.
- **EM** as P1b's:
  - the E-step gives each gold's responsibility over the nine sources;
  - the M-step is the softmax regression of W on those responsibilities, with an L2 penalty of 1e-3 (L-BFGS);
  - W starts at zero, and the loop stops at a relative gain below 1e-9 or after 200 iterations.

  The features are standardized over the fit questions, with every dataset weighted equally. Nothing is selected:
  the sources, the features, the penalty and the stopping rule are fixed here.

## 4. The call, fixed before any number

**The checks come first.** On every carve P1b read (s1sel on the five training datasets, s1eval on all six):
- **MX, P1b's filed gate,** is recomputed beside MXU from the same walks: P1b's six sources as section 2 maps them,
  P1b's features and P1b's gate. Its curve, and its values at |I_q| and |P4e|, must equal P1b's filed records at every
  point.
- **The zero-step sources must equal P1b's** at every point, and on the five graph datasets the regime's column of
  walks must equal P1b's S, L and R.

If any check fails, the stage stops: the comparison with MX would be unsound.

**Bug fix to the check, 13 October, about 08:45, before the verdict's numbers.** The first verdict run stopped at
metaqa's s1sel, on four points: S and L at |I_q| and at |P4e|. P2b's values there were about twice P1b's filed ones
(S at |I_q|: 0.8965 against 0.498). Every curve point agreed, and so did MX, the zero-step sources and R at every point.
- **The cause is in P1b's records, not in the walks.**
  - P1b's apply lists S and L twice: `DIAG = ("MX", "S", "L", "A1MIX") + SOURCES`, where SOURCES holds S and L. So each
    question's gold positions were appended twice to those two lists.
  - A curve is a mean over the list, and doubling every entry leaves it unchanged.
  - The at-size reading zips the list with the questions' sizes and divides by the list's length. So it reads the first
    half of the questions against shifted sizes and comes out at about half.
- **Shown exactly.** P1b's own walks were rebuilt on the laptop with P1b's own functions, on metaqa's s1sel:
  - read with each question once, they give P2b's four values;
  - read with the doubled list, they give P1b's filed four.
- **No P1b decision is touched.** P1b's call, its MX, its curves and every number in its file are unaffected, and P1b2
  never read those values. [P1B_RESTART_MIXTURE.md](P1B_RESTART_MIXTURE.md) section 7 carries the correction.
- **The fix.** The at-size points of P1b's S and L are now checked against P1a's A0 and LINKED.
  - These are the same walks: P1b's own check found them equal at every N.
  - P1a filed their values at the same two sizes, each question once.
  - P1b's filed values are kept in the verdict record beside them (`p1b_doubled_at_size`).
  - Every other comparison is unchanged, and a walk that differs from P1a's at either size still stops the stage (the
    self-test covers both).
  - The parts, the gate and the gathers were not rerun: the fix is in the verdict's check only.

**Per dataset, on s1eval,** MXU is compared with MX on the same questions:
- At every N of P1a's grid (25 to 7,000) and on both measures (ALL and gold recall):
  - **ABOVE** if MXU is more than 0.002 above MX somewhere and never more than 0.002 below;
  - **BELOW** if the reverse;
  - **MIXED** if both;
  - **TIE** if neither.
- At each question's size of today's pool (|I_q|), the difference on both measures. Today's pools and P1c's are built
  at that size.

**The call:**
- **ONE_RULE** if all three hold:
  - at |I_q| on s1eval, MXU is never more than 0.002 below MX, on either measure, on any of the six datasets;
  - it is more than 0.002 above MX on at least one;
  - on the five s1sel carves, MXU's mean ALL at |I_q| is not more than 0.002 below MX's.
- **REGIME_STAYS** otherwise. The record names each dataset and measure where MXU falls short.

**Reported beside, deciding nothing:**
- the curve labels above;
- MXU against the best LLM-free SOTA order at every N, with P1b's rule (AT_OR_ABOVE or BELOW, and the gaps);
- each of the nine sources' own curves. The st and all columns are the families' value under P1b's three restarts,
  beside P2a's A3, A1 and LINKED;
- the mean source weights per dataset, on the fit and on each read;
- the share of golds each source reaches on the fit carves.

## 5. What follows

- **ONE_RULE:** the per-dataset regime leaves the pools, and MXU is the construction the next pool rebuild reads in
  place of MX. That rebuild is declared in its own file after P1c's call, with P1c's screen: one fit per family, and a
  full run only if no read loses.
- **REGIME_STAYS:** the regimes stay. The shortfalls name the next change, in its own file.
- **No model's edges change here.** P2a part 2 found that the models read every family, with nothing to drop. P2b is
  the pools' walk only.
- **Information that is missing** (musique's links) is a later stage of P2, in its own file.

Flagged, as for every pool so far: the walks are propagation over the graph. The pool is built before either model
runs, and both models of a family read the same pool, so the MLP's own scoring stays free of message passing.

## 6. How it runs

- **Code:** `scripts/p2b_family_gate.py`. It imports, unchanged:
  - P1b's walk with masses, its zero-step sources, its features, its EM and its mixture;
  - P2a's family stores and step 4b's subset union;
  - step 4e's carve runner.
- **The self-test** covers:
  - each union's walks equal step 4e's walk on a union of those families alone;
  - P1b's six sources, rebuilt from the nine, equal P1b's own sources on the regime's union;
  - the features: the first nine equal P1b's, and the walk and degree features equal brute force;
  - a one-hot mixture reproduces each of the nine sources;
  - EM's log-likelihood never falls, and on a toy problem it picks each question's source among nine;
  - the verdict's labels and call come out as declared on made-up records, and the check refuses a recomputed MX
    that differs from P1b's.
- **Development runs, no number.** `--limit Q` runs gather or apply on the first Q questions of each carve and writes
  nothing. It logs only that every step ran, that the features are finite, and that P1b's six sources rebuilt from
  the nine equal P1b's own sources on the regime's union, array for array. Apply reads a uniform gate when none is
  fit. Before this file was committed, the self-test passed, and gather and apply ran with `--limit 300` on
  musique, metaqa and squad on the laptop: every step ran, the check held on every carve, and nothing was written.
- **Host jobs** (light lane), sized from P2a part 1's walks of the same stores (2wiki peaked at 4.6 GB on 8 CPUs):

  | item | what | where |
  | --- | --- | --- |
  | `p2b-pools-gather-<dataset>` (five) | features and the golds' masses on the fit carve | 2wiki 8 CPUs, 6 GB; hotpotqa 6, 5 GB; metaqa 4, 3 GB; musique 3, 2 GB; squad 3, 2 GB |
  | `p2b-pools-fit` | EM on the five gathers | 2 CPUs, 2 GB |
  | `p2b-pools-apply-<dataset>` (six) | MXU, MX and the nine sources on s1sel and s1eval | as gather; webqsp 3 CPUs, 3 GB |
  | `verdict` | the checks and the call | locally, on the fetched records |

  Records go to `outputs/p2b/`: the gathers are a few MB, and the reads' records are small.
- **ETA:**
  - P2a part 1 walked 2wiki's s1eval at about 17 ms per question and walk (8 threads).
  - gather: six walks on about 6,000 questions, about 12 minutes on 2wiki with loading;
  - fit: about a minute;
  - apply: 2wiki's 14,072 questions take about 25 minutes.

  The call should land about 45 minutes after the start, if the host's CPUs are free.
- The encoder stays frozen. No model trains or reads, nothing is adopted by this file, and test splits are never
  read.
