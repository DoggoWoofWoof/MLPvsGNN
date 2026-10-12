# P1b2: one restart for all six datasets, a gate fit to the golds' ranks over fixed blends (a diagnostic)

Declared 13 October 2026, before any number of the stage. P1's third stage in
[PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md), after [P1A_POOL_SCOREBOARD.md](P1A_POOL_SCOREBOARD.md) and
[P1B_RESTART_MIXTURE.md](P1B_RESTART_MIXTURE.md). Nothing here is adopted, and no model trains or reads.

## 1. What P1b left

P1b's mixture (MX) is above today's pools on all six datasets at the same size, but below the best LLM-free SOTA pool
order on five (P1b, section 7):
- metaqa under LINKED at 28 of 30 points;
- hotpotqa and webqsp under A1 (HippoRAG 2's restart rule) at the head;
- musique and 2wiki at one point each.

The sources were not short; the gate missed.
- metaqa's leading order is P1b's source L itself.
- The half-and-half mixture of P1b's S and R walks tracks A1 within 0.006 at every N shown.
- The gate was fit to the golds' mass, while the call reads their rank. EM moved weight to the walk that holds its mass
  near the seeds, and hedged against misses.

P1b2 keeps P1b's sources and features and changes two things:
- **The objective:** the gate is fit to the golds' ranks, the call's own measure, read on each question.
- **The read:** each question takes one fixed blend of the sources, so a question whose best order is a single source
  gets that source's order exactly. P1b's soft mixture moved every question part of the way.

HippoRAG 2's own walk joins as a seventh source, so the order that leads on four datasets is available exactly, not
only up to the walk's push threshold.

## 2. The construction (RC)

- **Seven sources per question**, on the dataset's regime union graph with step 4e's kernel (its alpha and eps),
  nothing excluded, each a walk's top 7,000 nodes with their masses scaled to sum to 1:
  - P1b's six: S0, L0 and R0 (the seeds, our linked nodes and the RRF lists, unwalked) and S, L and R (their walks);
  - A1: the walk from HippoRAG 2's restart (half the seeds, half the RRF lists), which is P1a's A1.

  squad has no graph. There S, L and R are S0, L0 and R0 (as in P1b), and A1 is P1a's A1 there: the RRF order, its
  weights scaled to sum to 1.
- **Eight classes**, each a fixed blend of the sources:

  | class | blend | is |
  | --- | --- | --- |
  | S | S | P1a's A0 |
  | L | L | P1a's LINKED |
  | R | R | the walk from the lists |
  | A1 | A1 | P1a's A1 |
  | LR | half L, half R | |
  | SL | half S, half L | |
  | SLR | a third each of S, L and R | |
  | MX | P1b's filed gate, per question | P1b's MX |

  A one-source class keeps its source's order. A blend orders by its blended mass. Ties go by node position, and
  every order is cut at 7,000.
- **The gate:** pi(q) = softmax(V x(q)) over the eight classes. x(q) is P1b's fifteen label-free features of the
  question's own lists, links and walks, unchanged. No feature reads a gold or names a dataset.
- **The pool at N** is the top N of the class the gate rates highest, among the classes whose order is not empty for
  the question (ties go to the earlier class in the table).
- **One parameter set for all six:** V, eight classes by sixteen weights (the fifteen features and a bias).

Reported beside, deciding nothing:
- **RCS, the soft read:** the blend sum over c of pi_c(q) times class c's blend, over the classes with a non-empty
  order.
- **ORC, the oracle:** per question, the class with the least area (section 3). It reads the golds, so it is an upper
  bound for any gate over these eight classes.
- every class's own curve.

## 3. The fit: EM to the golds' ranks, on the training datasets' fit carves only

- **Data:** P1b's fit carves of the five training datasets, whole (fit on metaqa, squad, hotpotqa and 2wiki; s1fit on
  musique). s1sel and s1eval are never fit. webqsp never fits, so its read is zero-shot.
- **A question's area under a class:** the sum over P1a's fifteen N (25 to 7,000) of (1 - ALL) + (1 - recall), with the
  question's own golds.
  - ALL is 1 when every gold is in the class's top N; recall is the share that is.
  - It is the call's measure, read on one question: 0 when every gold leads, 30 when no gold is held.
  - A question without golds has the same area under every class, so it carries nothing.
- **The evidence** for class c: e(q, c) = exp(-(area(q, c) - the question's least area) / tau), with tau = 1. A class
  one grid point behind the question's best keeps e^-1 of its evidence.
- **EM**, a mixture over the classes with the evidence as each class's likelihood:
  - E-step: each question's responsibility over the classes, pi_c(q) e(q, c) normalized;
  - M-step: the softmax regression of V on those responsibilities, with an L2 penalty of 1e-3 (L-BFGS, to
    convergence).
  - V starts at zero. The loop stops when the penalized log-likelihood gains less than 1e-9 relatively, or after 200
    iterations. This is P1b's EM code, with one evidence row per question.
- **Weights:** every question weighs the same within its dataset, and every dataset weighs the same.
- **Features are standardized** by their mean and spread over the fit questions, every dataset weighted equally. The
  questions and features are P1b's, so the values are P1b's.

Nothing is selected: tau, the classes, the penalty and the stopping rule are fixed here.

## 4. The call, fixed before any number

The checks come first. Any failure stops the stage.
- **The gather:** on every fit carve, the features equal P1b's gather, question by question.
- **The verdict, on every carve:**
  - class S equals P1a's A0, L equals LINKED and A1 equals A1, at every N on both measures (on squad, L and A1);
  - class MX equals P1b's MX.

For each dataset's s1eval carve, at every N of P1a's grid (25 to 7,000) and on both measures (ALL and gold recall),
RC is compared with the best of P1a's LLM-free SOTA orders (A1, A2, LINKED, DENSE, SPLADE, RRF) at that N:
- **AT_OR_ABOVE:** RC is never more than 0.002 below the best (step 4e's tie).
- **BELOW:** anything else. The record names every N and measure where RC is below, the order that leads there, and
  the largest gap.

Reported beside each call, deciding nothing:
- the same call for RCS and ORC;
- the points where RC is more than 0.002 above the best;
- RC, MX and A3 at the matched sizes of today's pools (|I_q|) and step 4e's (|P4e|), beside the best SOTA order and
  the sets;
- the share of questions each class takes under RC and under ORC, and the gate's mean weights;
- on metaqa and webqsp: RC at ReaRev's m (500, 2,000) beside the assigned-entity walk (ORACLE, a diagnostic), and the
  smallest N at which RC's ALL reaches ORACLE's at m.

s1sel is read the same way and reported beside.

## 5. What follows

- **All six AT_OR_ABOVE:** RC is the pool construction the models read next.
- **Otherwise:** the claim that our pools are at or above the SOTA pools on all six is not made. The pools step does
  not stall on it, because MX is already above today's pools on all six at the same size. The models read whichever
  of RC and MX has the higher mean ALL at |I_q| over the five training datasets' s1sel carves; ties go to MX.
  - Where ORC is AT_OR_ABOVE and RC is BELOW, the gap is the gate's.
  - Where ORC is BELOW too, no choice among these classes reaches the best order. The next change is a source or a
    class, declared in its own file.
- **P1c**, declared in its own file, rebuilds the models' pools from the chosen construction at each question's own
  |I_q|, so the look cost is unchanged, and runs one fit as a screen.

Flagged, as every pool so far: the walks are propagation over the graph. The pool is built before either model runs,
and both models of a family read the same pool, so the MLP's own scoring stays free of message passing.

## 6. How it runs

- **Code:** `scripts/p1b2_rank_gate.py`, built on P1b's script: its walks, features, gate and EM, imported unchanged.
  The self-test passes:
  - every blend's weights sum to 1;
  - the area is 0 when the golds lead, the whole grid when no gold is held, and counts the right N points between;
  - a one-source class keeps its source's order;
  - class MX equals P1b's mixture;
  - squad's A1 is P1a's RRF order;
  - the hard read skips empty classes, and the soft read blends only the classes it can read;
  - EM's log-likelihood never falls, and on a toy problem it picks each question's least-area class.
- **Development run, no number.** `--limit 64` runs gather and apply on the first 64 questions of one carve and writes
  nothing (apply reads a uniform gate). It logs only that every step ran and that the features equal P1b's. On the
  laptop, gather (squad fit, musique s1fit) and apply (squad and musique s1eval) ran through every step, and the
  features equal P1b's on both gathers.
- **Host jobs:** step 4f's pattern, light lane, sized from P1b's measured peaks: 5 CPUs; squad 1 CPU, 2 GB; the fit
  2 CPUs, 2 GB.

  | dataset | memory |
  | --- | ---: |
  | 2wiki | 5 GB |
  | hotpotqa | 4 GB |
  | metaqa | 3.5 GB |
  | webqsp | 3 GB |
  | musique | 2 GB |

  The steps:
  1. `gather` on each training dataset's fit carve writes `outputs/p1b2/gather_<dataset>.{npz,json}`: the features
     and the eight areas per question.
  2. `fit` writes `outputs/p1b2/gate.json` once all five gathers are in.
  3. `apply` on s1sel and s1eval writes `outputs/p1b2/rc_<dataset>.json`.
  4. `verdict`, run locally, writes `outputs/p1b2/verdict.{json,md}`. It refuses parts from different scripts, gates
     or freezes.
- **ETA:** P1b took 10 s to 7.7 minutes per gather, 28 s for the fit and up to 14.5 minutes per apply (2wiki). P1b2
  adds one walk and eight class areas per question. The call should land about 35 minutes after the start.
- The encoder stays frozen. No model trains or reads, nothing is adopted, and test splits are never read.
