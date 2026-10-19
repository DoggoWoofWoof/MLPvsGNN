# A6ex: the link words on the adopted arms, in each of zspx's steps and beside zrckpr's columns

Declared 19 October 2026, before any number. The next form A6e's verdict named for both arms
(docs/A6E_LINK_WORDS_EDGES.md, sections 5 and 7; `outputs/a6e/screen-verdict.json`), under docs/SCREENS.md section 3b
and docs/VERDICT_AUDIT_2026_10_17.md. Code: `outputs/mp_unified/za6ex.py`.

## 1. Why this, now

**A6e** (18 October) gave D4's link score to the lean pair, on the bases A6 used: eight split columns over the first
stage's top ten rows on zrckp (zrckpn), and edge weights in zsp's one step on zspk (zspkl). Both arms were NOT_ADOPTED
with no GAIN and no LOSS after the seed-null re-call (`outputs/a6e/screen-verdict.json`, 2026-10-18T21:16:55Z).

The reads that moved, from the re-calls (`outputs/screen/scr-a6e-zrckpn-pair-recall.json` and `-zspkl-`; delta = the
arm's R@5 minus its base's, with its 95% CI):

| arm | split | dataset | read | delta [95% CI] | floor | call |
| --- | --- | --- | --- | --- | ---: | --- |
| zrckpn | L-hotpotqa | hotpotqa | zero-shot | +0.0086 [+0.0055, +0.0116] | 0.0200 | filed GAIN, re-called WITHIN |
| zspkl | L-hotpotqa | hotpotqa | zero-shot | +0.0077 [+0.0046, +0.0109] | 0.0200 | filed GAIN, re-called WITHIN |
| zspkl | L-musique | musique | zero-shot | -0.0094 [-0.0149, -0.0042] | 0.0720 | filed LOSS, re-called WITHIN |
| zrckpn | L-hotpotqa | musique | in-domain | +0.0057 [+0.0009, +0.0108] | 0.0075 | WITHIN |
| zspkl | L-hotpotqa | musique | in-domain | +0.0071 [+0.0014, +0.0123] | 0.0075 | WITHIN |
| zrckpn | L-musique | hotpotqa | in-domain | +0.0054 [+0.0033, +0.0076] | 0.0075 | WITHIN |
| zspkl | L-hotpotqa | 2wiki | in-domain | +0.0070 [+0.0054, +0.0086] | 0.0075 | WITHIN |
| zspkl | L-musique | 2wiki | in-domain | -0.0052 [-0.0067, -0.0035] | 0.0075 | WITHIN |

So the words lifted hotpotqa on all four of A6e's fits (MLP +0.0054 and +0.0086, GNN +0.0013 and +0.0077), and
musique in domain on L-hotpotqa on both models, each within its floor. webqsp zero-shot, where all of A6's losses
were, stayed WITHIN on all four (MLP -0.0026 and -0.0067, GNN -0.0079 and +0.0078).

**The adopted arms.** A6 and A6e built on zrckp and zspk. Two newer forms of the lean pair passed their screens on
17 October:
- **G5's zspx** (zspk's step end to end, G3's residual message, G4's second step): ADOPT_FOR_FULL_RUN, with GAINs on
  L-hotpotqa's musique, +0.0147 [+0.0095, +0.0198] (0.5754 to 0.5901), and L-hotpotqa's webqsp, +0.0235
  [+0.0133, +0.0327] (0.2680 to 0.2914) (`outputs/screen/scr-g5-zspx-pair-recall.json`).
- **R1's zrckpr** (zrckp plus a pointwise head over nine fixed bridge-residual columns): ADOPT_FOR_FULL_RUN, with a GAIN
  on L-musique's webqsp, +0.0135 [+0.0032, +0.0239] (0.3627 to 0.3762)
  (`outputs/screen/scr-r1-zrckpr-pair-recall.json`).

zspx passes three messages where zspk passed one, and none of them reads a link's words. zrckpr carries R1's columns,
which A6e's columns would sit beside. A6e's section 5 fixed this form for no GAIN on both arms: the same inputs on the
adopted arms, on both models. The link-word idea closes as a model input only if A6ex finds no GAIN either.

## 2. The form

A6e's inputs, unchanged: the same build (`za6e.py build`), the same arrays under `outputs/a6e`, read through A6e's own
carve wrappers (`za6e.cols_carve`, `za6e.ew_carve`) around the adopted arms' carves. Only the bases change.

### The MLP (zrckprn)

- **The base:** zrckpr, which is zrckp plus R1's pointwise head over R1's nine columns (Linear(9, 16), GELU,
  Linear(16, 1)).
- **Added:** A6e's pointwise head over A6e's eight columns (A6e's section 2: log1p of the best title, relation and NER
  scores, the same scores over the question's total stem IDF, and two has-a-window flags, from the first stage's top ten
  rows by rrf, split by the graph's kind). Linear(8, 16), GELU, Linear(16, 1): 161 parameters, drawn from its own
  generator (seed + 6601), with the last layer zero at the start. Its output adds to zrckpr's score.
- **Two heads, side by side.** R1's head reads R1's nine columns; A6e's head reads A6e's eight. Neither reads the
  other's.
- **The carve:** zrckpr's (`zr1res.R1KTChainCarveZRC`: zrckp's with R1's columns beside, no edge), with A6e's columns
  beside.
- **The MLP rule.** Each row reads only its own columns. A6e's anchors come from rrf, before any row is scored. R1's
  columns are computed before either model runs. No model score enters a column, and nothing is normalised over the
  question's scores. zrckprn passes no message.

### The GNN (zspxl)

zspx (G5) adds three steps to s0, zspk's score before its step (zrc's scorer and the KT head). zspxl gives each step
A6e's six link-weighted inputs e(s), at the score that step reads:

    s1 = s0 + h_sp(prop_inputs_grad(s0), e(s0))    zspk's step, its inputs with the gradient kept (zspkg)   ew_w1
    s2 = s1 + h_rs(res_inputs(s1, c, s), e(s1))    G3's residual message, no gradient (zspr)                ew_rs_w1
    s3 = s2 + h_p2(prop_inputs(s2), e(s2))         G4's second step, no gradient (zspk2)                    ew_p2_w1

- **e(s)** is A6e's (`za6e.ew_inputs`). Per word family (title windows, relation names, shared entities) it holds the
  link-weighted mean of the row's neighbours' z-scores and log1p of the summed weight. It is 0 where the row has no
  worded neighbour of the family. The z is zsp's: the score s, detached, z-scored in its pool, clipped to ±8.
- **The weights** are A6e's: log1p of the pair's link score on each zlink edge, the same in both directions, and 0 on
  kNN edges.
- **Each step's head** is zspx's, with e entering its first layer through the step's own columns:
  h(x, e) = W2 · GELU(W1 x + b1 + E e) + b2. E is `ew_w1`, `ew_rs_w1` or `ew_p2_w1`, each 32 × 6 and zero at the
  start: 576 parameters in all. No random draw is added, so every base parameter starts as zspx's.
- **No new gradient path to the scorer.** e is computed from the detached score, so the end-to-end gradient of zspk's
  step is zspx's. At the start zspxl is zspx bit for bit: scores and every shared gradient. The new columns get a
  gradient from the first step on.
- **The carve:** zspx's (`zg3res.ResKTLinkCarveBase`: zspk's edges, with G3's c and s beside), with A6e's weights
  beside its edges.

**Which steps, and why all three.** A6e put the words where the GNN acts, on the edges: in zsp's one step, so that a
message along a worded link counts in proportion to its words (A6e's sections 1 and 2). zspx has three steps, and each
reads zlink's edges. None reads a link's words:
- zspk's step and G4's second step take a family's links alike (per family, the mean and soft-max of the neighbours'
  z, and log1p of the degree);
- G3's message weighs a link by its parent's score and reads its residual.

The words in one step only would leave two of zspx's three messages without them, a narrower form than A6e's. So each
step reads them. Each does so through its own columns, because the three heads are separate and each reads a
different score (s0, s1, s2).

**Message passing.** zspxl's steps pass messages, as zspx's do. The MLP passes none.

### Both

- **The arrays are A6e's.** No parameter, label or model score enters a column or a weight. The mentions are A6's
  (`outputs/a6/ents`). A6e's build checks its edges against zlink's, array for array, and its carves refuse weights
  built on other zlink edges.
- **Everything else is the base's:** the settings, seeds, batches, KT input, loss and loop. Each arm starts as its base
  bit for bit.
- **Serving cost.** As A6e's: the scores come from the pool's own text. The cost is a query-local compile, timed in its
  own file if an arm is adopted.

## 3. The arms

| arm | model | carve | its base |
| --- | --- | --- | --- |
| zrckprn | zrckpr's model plus A6e's head over the eight columns (MRO: ZRCKPRN, ZRCKPR, ZRCKP, ...) | zrckpr's, with A6e's columns beside (`A6eR1KTChainCarveZRC`) | `scr-r1-zrckpr`, `-hp` |
| zspxl | zspx's model with A6e's six inputs in each of its three steps (MRO: ZSPXL, ZSPX, ZSPR, ZSPK, ZProp, ...) | zspx's, with A6e's weights beside its edges (`EWResKTLinkCarveBase`) | `scr-g5-zspx`, `-hp` |

## 4. The fits and the call

A6e's, on the adopted bases:
- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-a6ex-<arm>`, `-hp`), on P1c2's roots with
  P1c3's KT input, at the card fits' settings (variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128). zspxl reads
  G3's builds and zrckprn reads R1's columns, as their bases do. Both read A6e's arrays.
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split (zrckprn: scr-r1-zrckpr and -hp; zspxl:
  scr-g5-zspx and -hp), with step 1's fit beside.
- **The re-call's base screens:** each base's card fit against step 1's. For zspx they are A9's
  (`outputs/a9/base-zspx-<split>.json`). For zrckpr they are this stage's own (`outputs/a6ex/base-zrckpr-<split>.json`).
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), per arm. GAIN,
  LOSS and WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A6ex's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b) and names each arm's next form (section 5). It also records
the link-word idea's state as a model input:
- **OPEN** while a complete re-call has a GAIN;
- **WAITING** while a re-call is incomplete and no complete one has a GAIN;
- **CLOSED_AS_A_MODEL_INPUT** when both re-calls are complete with no GAIN.

**Reported beside, deciding nothing:**
- musique (D4's target, the GNN base's GAIN), webqsp (A6's losses, both bases' GAINs) and hotpotqa (A6e's rises);
- musique in domain (L-hotpotqa, Claim 1's read), both arms side by side;
- hit@1 and FC@5 per read.

When the results are filed, A6e's deltas (zrckpn, zspkl) are set beside each read.

## 5. What each outcome leads to (fixed now)

Per arm. Without a GAIN of its own, an arm's next form also depends on the other arm's re-call: an idea with a signal
anywhere reaches both models as an input before it closes (SCREENS.md section 3b).

| this arm's re-call | the other arm's re-call | the next form |
| --- | --- | --- |
| ADOPT_FOR_FULL_RUN | any | the arm fitted again on P1g-b's pools (A6e's columns and weights built again on P1g-b's looks), then its full run, its own declared file |
| NOT_ADOPTED, with a GAIN (so with a LOSS) | any | **A6exp**, the passage-graph back-off on the adopted arms: the same arms with the typed graphs' relation-name words left out (the relation columns and weights 0), so only title windows and shared entities move either model; on both models, its own declared file |
| NOT_ADOPTED, no GAIN | ADOPT_FOR_FULL_RUN | zspxl: **A6exc**, A6e's eight columns on zspx as well, through A6e's pointwise head added to the score before zspx's first step reads it (where A6's and R1's GNN arms put their heads), so the GNN reads the input the MLP gained with. zrckprn: **A6exw**, the part of the edge weights the MLP may read without a message: per word family, log1p of the row's summed link weight over its zlink edges (A6e's weights, no score of any row), three columns beside zrckpr's through A6e's pointwise head at three inputs. Each in its own declared file |
| NOT_ADOPTED, no GAIN | NOT_ADOPTED, with a GAIN | **A6exp**, as above |
| NOT_ADOPTED, no GAIN | NOT_ADOPTED, no GAIN | the link-word idea closes as a model input (below) |
| NOT_ADOPTED, no GAIN | INCOMPLETE | nothing closes until the other arm's missing re-call is run; then this table applies |
| INCOMPLETE | any | the missing re-call is run; nothing else changes |

**Why these.**
- **A GAIN with a LOSS.** All of A6's losses were on webqsp zero-shot, where the words are a KB's relation names. A6e
  fixed this back-off (A6ep) for the same outcome. A6exp is it on the adopted arms.
- **No GAIN here, but the other model's arm adopted.** The input that moved one model has not yet reached the other in
  the form that moved it. A6exc gives the GNN the columns the MLP gained with. A6exw gives the MLP the part of the
  weights its rule allows: each row's own summed weights per family. They come from the pool's text alone, no score of
  any row enters them, and no message passes.
- **No GAIN on either arm.** The idea closes as a model input. A6, A6e and A6ex gave the link words to both models as
  learned inputs that start at zero: columns on the MLP; on the GNN, columns, then edge weights in its one step, then
  edge weights in every step that reads edges. They did so on the old bases and on the adopted ones, and no re-call
  found a GAIN. D4 stays a diagnostic. This is the close A6e's section 5 fixed.

No model change is adopted on today's pools (the user's rule of 13 October): an adopt is refitted on P1g-b's pools
before its full run.

## 6. How it runs

**Self-test.** `python outputs/mp_unified/za6ex.py --selftest` passes before this commit (7.5 s). It builds one toy look
with every array the arms read: rmatch's typed toy look, G3's toy pools and embeddings over A6's toy corpus, an rrf
column, then rmatch's chains, zlink's edges, G3's residual parts, R1's columns, and A6e's columns and weights (read once
as a passage graph and once as a KB), each by its own builder. It checks:
- **the arms:** the MROs; the bases and A6e's arms unchanged; 161 and 576 added parameters;
- **the batches:** every value is the base carve's bit for bit, with A6e's columns or weights beside (equal to A6e's own
  carves'). zrckprn's batch holds no edge, no weight and no residual message. The edges' word families are right on
  each reading;
- **the models:**
  - from the same seed, each arm is its base's state plus its added parameters, with no draw moved;
  - with the base's own heads acting (R1's head; zspx's three steps), each arm starts as its base bit for bit, in eval
    and in training: scores and every shared gradient. The added parameters get a gradient at once;
  - zrckprn: A6e's head moves the score by exactly its term, and one row's columns move that row's score only;
  - zspxl: the six inputs equal a loop over the edges and carry no gradient to the scorer. Each step's columns move
    that step only, against zspx's chain rebuilt from G5's parts; all three together; and without G3's c and s;
  - without its side key each arm is its base;
- **toy fits** through lean_gpu's loop repeat bit for bit, and the added parameters learn. On the base's carve (no side
  key) every state is the base fit's bit for bit;
- the train and read refusals; relz's pair and re-call per arm, stamped A6ex's; the verdict's outcomes, next forms and
  the idea's state.

**Host items** (19, `a6ex-*`):
- **No rebuild.** A6e's arrays are on the host as A6e built them: on 19 October all 78 (the columns and the weights of
  39 parts) matched the sha256 in their records. While A6ex's items are queued the host's clean-up keeps them
  (za6ex.py imports za6e.py, which writes `outputs/a6e`), and A6e's carves check each array against its record when
  they read it. A6e's records stay as filed.
- **Base screens:** 2 (`a6ex-base-zrckpr`, `-hp`): zrckpr's card fits against step 1's, with no dependency. P1g-c's
  `p1gc-base-pools-zrckpr-*` compute the same compare into `outputs/p1gc/`; A6ex's re-call reads its own.
- **The screen:** 17 items. Per arm: two trains, two reads, two compares, a pair and a re-call. Then the verdict.
  - **zspxl** has A9's sizes for zspx plus A6e's increment for the per-edge weights (A6e gave zspkl 0.03 more of the
    card and 1 to 1.5 GB more than A6's zspkw): `--frac 0.43`, gpu 0.45, memory 7 GB (L-musique train), 16.5 GB
    (L-hotpotqa train) and 7.5 GB (reads). Its trains and reads follow G3's rebuilt parts (`n19-res-*`, by A9's carve
    lists).
  - **zrckprn** has R1's sizes: `--frac 0.30`, gpu 0.32, memory 4.5 GB (trains) and 6 GB (reads). Its trains and reads
    follow R1's rebuilt columns (`n19-r1c-*`, by R1's carve lists).
  - zspxl's re-call follows A9's base screens (`a9-base-zspx`, `-hp`). zrckprn's follows `a6ex-base-zrckpr`, `-hp`.
    The verdict follows both re-calls.
- **inputs:** A6e's list, plus zg3res.py, zg4prop.py, zg5mix.py, zr1res.py and za6ex.py.

**The commands.** One arm and split are shown. zrckprn and L-hotpotqa (names ending `-hp`) follow the same pattern. On
the host the trains and reads run under `retry_cmd.py --cuda` and `cuda_alloc.py --frac`, as in the items.

    python outputs/mp_unified/za6ex.py compare --new outputs/screen/fits/scr-r1-zrckpr \
        --base outputs/step1/fits/L-musique --out outputs/a6ex/base-zrckpr-L-musique
    python outputs/mp_unified/za6ex.py train --split L-musique --name scr-a6ex-zspxl --arm zspxl --device cuda \
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za6ex.py read --name scr-a6ex-zspxl --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za6ex.py compare --new outputs/screen/fits/scr-a6ex-zspxl \
        --base outputs/screen/fits/scr-g5-zspx,outputs/step1/fits/L-musique --out outputs/screen/scr-a6ex-zspxl
    python outputs/mp_unified/za6ex.py pair --arm zspxl \
        --screens outputs/screen/scr-a6ex-zspxl.json,outputs/screen/scr-a6ex-zspxl-hp.json \
        --out outputs/screen/scr-a6ex-zspxl-pair
    python outputs/mp_unified/za6ex.py recall --arm zspxl \
        --null outputs/screen/scr-null-s1.json,outputs/screen/scr-null-s2.json,outputs/screen/scr-null-s1-hp.json,outputs/screen/scr-null-s2-hp.json \
        --pair outputs/screen/scr-a6ex-zspxl-pair.json --out outputs/screen/scr-a6ex-zspxl-pair-recall
    python outputs/mp_unified/za6ex.py verdict

## 7. Results

Not run yet.
