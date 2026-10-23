# U2h: W2_S beside today's graph, as an input to both models

Declared 23 October 2026, after U2g's verdict and before any U2h number. Code: `outputs/mp_unified/zu2h.py`, and U2h's
stages added to `outputs/mp_unified/zu2d_modal.py` in a follow-up commit, as U2g's were. Everything else is imported
unchanged: zu2g.py, zu2d_pools.py, zu2d.py, scripts/p1c3_kt.py, zp1c3, zp1c3p and lean_cache. Numbers here are
development numbers.

## 1. The question

U2g's cell Y (docs/U2G_GRAPH_AND_RULE.md, section 10) put W2_S, the LLM-free links, in place of today's graph, at
P1c2's pool rule:
- both lean models gained musique on L-musique;
- both lost hotpotqa on every read, and 2wiki on three of four.

docs/SCREENS.md (3b), the user's rule of 17 October: "Gains on some datasets and a loss on others call for a form that
can back off, per question or per dataset (an input, a gate), never a stop." An idea closes only when its signal has
reached both models as an input that each model is free to use or not.

U2h is that form. Today's graph and P1c2's pools stay exactly as they are: U2d's base build, unchanged. W2_S comes in
beside them as one more input, P1c3's KT columns computed on W2_S. Each model learns how much to use it.

## 2. The input: KT_W

**What it is.** P1c3's eight columns (scripts/p1c3_kt.py's `columns`): KIND_T, A1st, A1all, Kst, KTst, KTall, fill
and entity. They are computed from Y's orders, for the rows of U2d's base build.
- **Y's orders:** zu2g's carve opener (W2_S swapped in, P1c2's rule), then zu2d_pools.question, unchanged. These are
  the orders Y's pools stage read.
- **The rows:** each question's pool on the base build (B and P1c2's expansion on today's graph), in the base caches'
  row order.
- **fill:** 1 where the row is in Y's expansion, the fill of KIND_T's order on W2_S. **entity:** Y's kind of the
  question (P1e's).

So for a base row, KT says where today's walks place it, and KT_W where W2_S's walks place it.

**Where.** squad, musique, hotpotqa and 2wiki, on P1c2's carves: fit or s1fit, and s1eval. metaqa and webqsp run on
the KB in both builds, so their KT_W is base's KT, byte for byte.

**The checks.** Each one stops the stage:
- per question, B is base's and Y's, and Y's expansion is Y's filed one;
- Y's own rows, from the recomputed orders, give Y's filed KT columns bit for bit, so the orders are Y's;
- each question's cache rows are its base pool, each node once (zu2d_pools.to_cache_order);
- the questions, their order and the rows per question are base's KT record's.

## 3. The arms

U2d's via arms (docs/U2D_REFIT_ON_U2E.md, section 4), with KT_W in place of via:

| arm | model | head inputs | parameters |
| --- | --- | --- | --- |
| zrckpw | zrc's, plus zrckp's pointwise head widened | the row's KT and KT_W columns (16) | 289 |
| zspkw | zsp's, plus zspk's head widened, before zsp's propagation reads the score | the row's KT and KT_W columns and its z-scored score (17) | 305 |

- The head is one hidden layer of 16 (GELU) and one output added to the score.
- The MLP's head reads no score, as zrckp's does not.
- In both arms the first layer comes from its own generator (seed + 3301, P1c3's) and the last layer is zero. At the
  start, each arm's forward is its base arm's (zrckp's, zspk's) bit for bit.
- za6lw.py gives two of its own arms the names zrckpw and zspkw. zu2h.py stops if lean_screen already holds other arms
  under these names, so the two files never share a process.

## 4. The fits

Four fits, on U2d's card:
- seed 0, on L-musique and on L-hotpotqa;
- P1c2's card on the GPU, every carve on U2d's base build;
- read with p@swa on the six s1eval carves;
- on Modal's L40S, from U2d's image (u2d-gpu:py3.13:torch2.8.0+cu128:numpy2.3.2).

Their names are scr-u2h-zrckpw, scr-u2h-zrckpw-hp, scr-u2h-zspkw and scr-u2h-zspkw-hp.

The base build in workspace 4 is a byte-for-byte copy of the one in workspace 3, which U2d's base fits read. Each
file's sha256 is taken at the download and checked again in workspace 4. So U2h's fits read the base fits' data.

## 5. The decision (H)

Each arm is compared with its base arm's seed-0 fit on the base build, per split (scr-u2d-base-zrckp[-hp] and
scr-u2d-base-zspk[-hp]). The two splits are paired, and the pair is re-called under U2d's seed null of the base arm
(seeds 1 and 2 on the base build; docs/U2D_REFIT_ON_U2E.md, section 6). zrckpw's null calls zrckp; zspkw's calls zspk.
The pair base is the null's own seed-0 fit, so nothing is re-based.

GAIN, LOSS and WITHIN are docs/SCREENS.md's, as in U2d and U2g. A GAIN needs delta R@5 ≥ 0.0075 with the 2,000-resample
bootstrap's 95% interval above 0. The re-call turns a GAIN or LOSS below the read's floor into WITHIN.

**The rule, fixed before any number.** Over both arms and their twelve reads each:
- **ADOPT_FOR_FULL_RUN:** no LOSS on any read, and at least one GAIN;
- **NO_LOSS:** no LOSS on any read, and no GAIN;
- **NOT_ADOPTED:** at least one LOSS;
- **INCOMPLETE:** a re-call is missing, is not U2h's, does not hold twelve reads, or compares fits from different
  devices.

## 6. Reported beside, deciding nothing

- **U2h against Y:** each U2h fit against Y's fit of the same base arm and split (`scr-u2h-vy-<arm>[-hp]`). That is W2_S
  as an input beside today's graph, against W2_S in place of it. These are readings: the two fits ran in different
  workspaces on different builds.
- **Per carve:** the share of base rows in KIND_T's order on each graph, and how often their places differ.
- **zrckpw's share** of zspkw's gain over RRF.
- FC@5 and hit@1 for every read, and hit@1 as the headline on metaqa and webqsp.

## 7. What follows (fixed now)

Each verdict closes only the form it tests (docs/SCREENS.md, 3b).
- **ADOPT_FOR_FULL_RUN:** the lean pair's full run with KT_W (six splits per arm, under grade-nullx), declared in its
  own file, with zg6 on the same input (zg6w) beside it.
- **NO_LOSS:** the rest of W2_S's signal, as the next input: the look block's graph columns computed on W2_S for the
  base rows (topo_STRUCT, depth_STRUCT, walk, walkf and dists), beside today's, in its own file.
- **NOT_ADOPTED:** a per-question gate on the KT_W head, its weight per question taken from inputs computed before any
  row is scored, in its own file.
- **In every case,** U2g's section 7 stands: the links chosen by the lean pair's R@5, then X, when credit or the host
  allows.

## 8. How it runs

On Modal, in workspace 4. Workspace 3 holds the base build but has spent its credit.
- **The copy.** From workspace 3, through the laptop, byte for byte:
  - the base build's fit inputs: every file under `/out/base` except the looks' arrays (1,685 files, 25.96 GB);
  - Y's pools: its parts, manifests and same_as_new records under `/out/y/pools`.
- **`--stage h-ktw`:** KT_W in shards, the same shards as Y's pools (59). Then each carve's merge into the base
  caches' row order, and metaqa's and webqsp's from base's KT.
- **`--stage h-fits`:** the four fits, each training and reading in one L40S container.
- **`--stage fetch-h` and `--stage fetch-h-fits`:** the records and reads come to the laptop.

Then, on the laptop: `python outputs/mp_unified/zu2h.py compare-all`, then `decide`.

**The cost:**
- KT_W: about $1. It walks what Y's pools stage walked, and that cost under $1.
- The four fits: about $2.2.

Workspace 4 has about $10 left.

## 9. Results

Filed after the run.
