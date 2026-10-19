# A9b: the MP-unified track's bound alone on the adopted GNN

Declared 20 October 2026, before any number. The next form A9t's verdict named for no GAIN
(docs/A9T_BOUNDED_DROPOUT.md, section 5; outputs/a9t/screen-verdict.json), under docs/SCREENS.md section 3b. Code:
`outputs/mp_unified/za9b.py` (the arm, the screen and the call; it imports A9t's bound from `za9t.py` and changes no
frozen module). Not run yet.

## 1. Why this, now

**What A9t found.** zspxdt was filed NOT_ADOPTED on 20 October: 0 GAIN, 1 LOSS. zspxdt is zspx, the adopted GNN,
trained with A9's edge-family dropout (in training each question drops each of its link families, zlink's three kinds
of link edge, with probability 0.25) and with zspx's three increments bounded together at 0.5. zspx is G5's arm: zspk
(the GNN with P1c3's KT input) with its message-passing step trained end to end, then G3's residual message and G4's
second step, three steps over the question's link edges.

A9t's re-call under the seed null calls twelve reads, the six datasets read from each of the two screen splits
(L-musique and L-hotpotqa: each leaves that dataset out of training, and webqsp never trains), each against a floor set
by how much fits differ by their seed alone. It found:
- **the LOSS:** musique in domain on L-hotpotqa, the read Claim 1 uses for musique (Claim 1: the MLP keeps at least 90%
  of the GNN's gain; docs/PAPER_CLAIMS_2026_10_09.md). R@5 (the share of a question's gold passages in its top 5) fell
  −0.0125 [−0.0177, −0.0074] against its 0.0075 floor, and FC@5 (all of a question's gold passages in the top 5)
  −0.0219;
- **webqsp zero-shot on L-hotpotqa:** −0.0208 [−0.0287, −0.0131], a LOSS by its pair's interval and WITHIN under its
  0.0233 floor;
- **no read up past its floor.** The largest rise was squad in domain on L-musique, +0.0035 [+0.0015, +0.0054] (floor
  0.0075).

**What A9 found.** zspxd, zspx with the dropout alone, was filed NOT_ADOPTED on 19 October: 0 GAIN, 0 LOSS, all twelve
reads WITHIN.

**What the bound changed beside the dropout.** A9 and A9t differ only in the bound. On the five reads where they differ
most:

| read | zspx R@5 | zspxd: dropout alone (A9) | zspxdt: dropout and bound (A9t) |
| --- | ---: | --- | --- |
| L-hotpotqa musique (in domain) | 0.5901 | −0.0059 [−0.0108, −0.0012] | −0.0125 [−0.0177, −0.0074] |
| L-hotpotqa webqsp (zero-shot) | 0.2914 | −0.0037 [−0.0121, +0.0043] | −0.0208 [−0.0287, −0.0131] |
| L-hotpotqa hotpotqa (zero-shot) | 0.8547 | −0.0075 [−0.0102, −0.0046] | −0.0035 [−0.0064, −0.0006] |
| L-hotpotqa 2wiki (in domain) | 0.9312 | +0.0026 [+0.0012, +0.0041] | −0.0047 [−0.0062, −0.0032] |
| L-musique musique (zero-shot) | 0.5073 | +0.0024 [−0.0036, +0.0079] | −0.0012 [−0.0077, +0.0049] |

Each difference is R@5 against zspx's card fit of the same split, from the stage's own re-call (section 7 of
docs/A9_EDGE_FAMILY_DROPOUT.md and of docs/A9T_BOUNDED_DROPOUT.md). With the bound added, the two L-hotpotqa reads
where G5's zspx had gained on zspk (musique +0.0147 and webqsp zero-shot +0.0235, both GAINs; docs/G5_COMBINED_GNN.md,
section 5) fell further, 2wiki in domain and musique zero-shot turned from up to down, and hotpotqa zero-shot fell
less. On zspx the bound has acted only together with the dropout, so these numbers cannot say what the bound does
alone.

**Where the bound comes from.** The MP-unified track (early October; docs/STATUS_2026_10_05.md) trained one-graph
message passing in its QD-GNN scripts. There qd_gnn13 added the dropout (`-fd25`) to arms that already bounded what
message passing adds to the score before it (`-tb5`: s = z + 0.5 tanh((s_mp − z) / 0.5), qd_gnn7's `-tb` rule).
Unbounded message passing on one graph hurt zero-shot reads there. G2 (15 October) carried both, but over its own
stack's readout on zspk (0 GAIN, 0 LOSS). On zspx's three steps the bound has acted only with the dropout (A9t), never
alone.

**Why this, and what it closes.** Under section 3b a verdict closes only the form it tested, and each verdict names the
next form or says why none is left. The track's form has two parts and three ways to put them on the adopted GNN: the
dropout alone (A9), both together (A9t) and the bound alone (this stage). A9t's section 5 named this stage for no GAIN.
It is the last of the three; section 5 below fixes what follows each outcome.

## 2. The form

zspxt is zspx (outputs/mp_unified/zg5mix.py) with one change: the score after zspx's three steps is bounded around the
score before the first, exactly as A9t bounds it. Nothing of A9's dropout is kept.

**Where s0 and s3 are taken** (zspx's forward, `zg5mix.ZSPX.forward`, lines 93 to 101), as in A9t:
- **s0** is line 94: the score of zspk's scorer and KT head (P1c3's input of KIND_T's walk positions), before any
  step. It reads no edge.
- s1 is line 98: zspk's step, its gradient kept.
- s2 is line 100: G3's residual message (on a carve with G3's residuals).
- **s3** is line 101: the score zspx's forward returns, after G4's second step.

**The change.** zspxt returns

    out = s0 + 0.5 tanh((s3 − s0) / 0.5)

in place of s3. The three increments are bounded together, not each. Each step still reads the unbounded score before
it (s0, s1, s2), as in zspx; the bound acts only on what the forward returns. Without links (lines 95 and 96) the
forward returns s0, and there is nothing to bound.

**The width.** 0.5, A9t's: qd_gnn13's tb5 (qd_gnn7's `-tb<T>` rule, t = T / 10), the width G2 used (zg2.TB). A move
s3 − s0 of 0.1 becomes 0.0987; a move of 1.0 becomes 0.482; no move passes 0.5. The track's EM arms also ran a 0.2
width (qd_gnn14's tb2); that width is not this form, as it was not A9t's.

**The gradient.** It is kept through s0 and s3; nothing is detached. With x = (s3 − s0) / 0.5,
∂out/∂θ = (1 − sech² x) ∂s0/∂θ + sech² x ∂s3/∂θ. The scorer keeps its full path; the steps' increments get sech² x of
theirs, which is 1 where the move is small.

**When.** In training and at read.

**No dropout.** Every step reads every link edge of the question, in training and at read, as in zspx. No family is
dropped, no generator is made and no draw is taken: zspxt's random draws are zspx's, draw for draw. A9's rate
(zg2.FD, 0.25) never reaches zspxt.

**Code.** `za9t.BoundedSteps` is A9t's class: zspx's forward restated line for line (zg5mix is frozen), s0 kept, and
s0 + TB tanh((s3 − s0) / TB) returned, with TB A9t's width (`za9t.TB`, 0.5). A9t's arm put zspxd's dropout over it
(`ZSPXDT`, whose classes in Python's order are ZSPXD, BoundedSteps, ZSPX). `za9b.ZSPXT` is BoundedSteps with nothing
added and zspxd left out (BoundedSteps, ZSPX). The train and read commands refuse to run unless the width is 0.5.

**What does not change.** No parameter is added: zspxt has zspx's three step heads, 385 parameters each. No new graph,
column, encoder or text. Settings, seeds, batches, KT input, loss and loop are zspx's.

**Bit-for-bit anchors.** At the start every step head's last layer is zero, so s3 = s0 and zspxt is zspx. With the
bound taken off, which stands for an infinite width (in floating point an infinite width would compute ∞ · 0), zspxt
is zspx, scores and gradients; at a width of 10⁶ it is zspx to float32 rounding. At width 0.5 it is A9t's zspxdt with
the dropout's rate at zero, in training and at read, scores and gradients; at read, where zspxdt draws nothing, it is
zspxdt at any rate.

**GNN only.** The MLP takes no message, so it has nothing to bound.

## 3. The arm

The carve is what a model reads for a split: the questions' pool rows and their inputs.

| arm | the model | the carve |
| --- | --- | --- |
| zspxt | zspx's, its three increments bounded together at 0.5 (A9t's bound), every link edge read in training and at read | zspx's (G3's: zspk's carve with c, each row's cosine to the question, and s, each link edge's cosine between its two ends, beside) |

## 4. The fits and the call

- **Fits:** one fit on the host's GPU (a card fit) on L-musique and one on L-hotpotqa (`scr-a9b-zspxt`, `-hp`), on
  P1c2's roots with P1c3's KT input. Card fit settings: variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128
  (zspx's).
- **Reads:** each fit on the six datasets' s1eval carves (the screens' evaluation questions) at p@swa (the fit's SWA
  state, its weights averaged over epochs).
- **Compares:** each fit against zspx's card fit of the same split (scr-g5-zspx, scr-g5-zspx-hp), which decides.
  Step 1's fit is reported beside.
- **The re-call's base screens:** A9's, zspx's card fits compared with step 1's of the same split
  (outputs/a9/base-zspx-<split>).
- **Call:** relz's pair and its re-call (outputs/mp_unified/relz.py: the two splits' compares joined into one call of
  twelve reads, then each read called against its seed-null floor, with the four null files of docs/SCREENS.md).
  GAIN, LOSS and WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

- **ADOPT_FOR_FULL_RUN:** zspxt's re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A9b's re-call of zspxt, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- webqsp and metaqa, both splits (the knowledge-base graphs: webqsp is the zero-shot one);
- musique in domain (L-hotpotqa): Claim 1's read, and A9t's LOSS;
- hit@1 and FC@5 per read, as the compares give them;
- zspxdt's difference per read from A9t's own re-call (the dropout and the bound) and zspxd's from A9's (the dropout
  alone), so the three ways of putting the track's form on zspx can be read side by side.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | zspxt fitted again on P1g-b's pools, then its full run beside zspx's (its own declared file) |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A9bq, the learned back-off: the bound's width per question, 0.5 · 2σ(a · m_q + b), m_q the shares of the question's link edges in the three families; a and b start at zero, so A9bq starts as A9b; no dropout (one declared file) |
| NOT_ADOPTED, with no GAIN | none of this track: the MP-unified track's form closes on the GNN. Its dropout (A9), its bound (A9b) and both together (A9t) will all have reached the adopted GNN without a GAIN, and the MLP takes no message, so the idea has reached the one model that can take it. This closing is the step the verdict records; no further form of this track is declared |
| INCOMPLETE | the missing re-call is run; nothing else changes |

No model change is adopted on today's pools (the user's rule of 13 October).

## 6. How it runs

**Self-test, passed before this commit.** `python outputs/mp_unified/za9b.py --selftest` (13 to 25 s) checks, on a toy carve
with G3's residuals built as A9t's:
- **The arm.** A9t's bounded steps over zspx with nothing added, on G3's carve: zspxd (the dropout) is not among its
  classes, and it has no method of its own. zspxdt's, zspxd's, zspx's and zspk's arms are unchanged. The width and the
  bound's settings are A9t's (0.5), the rate A9's (0.25 over three families, never read by zspxt); A9's base fits and
  base screens are the re-call's.
- **Bit-for-bit cases.**
  - zspxt's state equals zspx's and zspxdt's from the same seed. Building it moves the global random stream as zspx's
    does, and it has no generator.
  - At the start, zspxt is zspx, and both are zspx's scorer (s3 = s0).
  - With all three step heads moved, small (every move below 0.05) and large (the bound takes more than 0.5 off some
    score), at read: with the bound off zspxt is zspx, scores and every gradient; at a width of 10⁶ it is zspx to
    float32 rounding (gradients within 10⁻⁴); at 0.5 it is the bound of s0 (zspx's scorer) and s3 (zspx's score), and
    it is zspxdt, scores and every gradient, with its three heads' gradients different from zspx's.
  - In training, zspxt is the bound over zspx on every edge, its random draws zspx's. Setting A9's rate to 1 (each
    question would keep one family) moves zspxdt and leaves zspxt unchanged; at rate 0, zspxdt is zspxt, scores and
    every gradient; with the bound off, zspxt's training gradients are zspx's.
- **Toy fits through lean_gpu's training loop.** A repeat is bit for bit; the bound moves the fit off zspx's; zspxdt's
  fit at rate 0 is zspxt's bit for bit, and zspxt's fit at rate 1 is its own; with the bound off, and without links,
  the fit is zspx's bit for bit.
- **The rest.** Train refuses another arm (zspxdt, zspxd, zspx), split, fit name, cache root, device or width; read
  refuses a fit that is not A9b's, and any fit when the width is not 0.5. relz's pair and re-call are stamped A9b's,
  with relz restored after. The verdict's outcomes and next forms, with zspxdt's and zspxd's reads beside from A9t's
  and A9's re-calls (another arm's re-call at A9t's path is not taken beside), and six kinds of INCOMPLETE: a missing
  re-call, A9t's, A9's, another arm's, another declared file's, and one without the twelve reads. Nothing is written
  under the stages' folders (outputs/a9b, a9t, a9, g5 and screen).

**Commands.**

    python outputs/mp_unified/za9b.py train --split L-musique --name scr-a9b-zspxt --arm zspxt --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za9b.py train --split L-hotpotqa --name scr-a9b-zspxt-hp --arm zspxt --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za9b.py read --name scr-a9b-zspxt --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za9b.py read --name scr-a9b-zspxt-hp --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za9b.py compare --new outputs/screen/fits/scr-a9b-zspxt --base outputs/screen/fits/scr-g5-zspx,outputs/step1/fits/L-musique --out outputs/screen/scr-a9b-zspxt
    python outputs/mp_unified/za9b.py compare --new outputs/screen/fits/scr-a9b-zspxt-hp --base outputs/screen/fits/scr-g5-zspx-hp,outputs/step1/fits/L-hotpotqa --out outputs/screen/scr-a9b-zspxt-hp
    python outputs/mp_unified/za9b.py pair --screens outputs/screen/scr-a9b-zspxt.json,outputs/screen/scr-a9b-zspxt-hp.json --out outputs/screen/scr-a9b-zspxt-pair
    python outputs/mp_unified/za9b.py recall --null outputs/screen/scr-null-s1.json,outputs/screen/scr-null-s2.json,outputs/screen/scr-null-s1-hp.json,outputs/screen/scr-null-s2-hp.json --pair outputs/screen/scr-a9b-zspxt-pair.json --out outputs/screen/scr-a9b-zspxt-pair-recall
    python outputs/mp_unified/za9b.py verdict

The verdict writes `outputs/a9b/screen-verdict.{json,md}`. On the host the trains and reads run under `retry_cmd.py
--cuda` and `cuda_alloc.py`, as A9t's.

**Host items** (`a9b-*`, 9 items, A9t's 9 renamed):
- **On the card:** the two trains at once, then the two reads, sized as A9t's (A9's, from zspx's runs): `--frac 0.40`
  (GPU share 0.42) and 1.1 CPUs each. zspxt reads every edge in training, as zspx does.
- **Memory:** each item asks for its A9t counterpart's measured peak plus 0.5 GB, plus the feeder's 15%
  (`@a9t-train-zspxdt*1+0.5` and so on). A9t's items had asked for 6 GB for each train and read (15 GB for the
  L-hotpotqa train) and 4 or 2 GB for the CPU items. A9t dropped edges in training and zspxt does not; the 0.5 GB is
  meant to cover that (an estimate).
- **Waits:** the trains wait on the rebuild of G3's residual arrays (the `n19-res-*` items) for their split's fit
  carves, 11 for L-musique and 16 for L-hotpotqa; each read waits on its train and the six s1eval carves' 22.
- **On the CPU** (1 CPU each): 2 compares, the pair, the re-call (after A9's base compares, `a9-base-zspx` and `-hp`)
  and the verdict.
- **Inputs:** A9t's list with `za9b.py` added; `za9t.py` stays, since za9b imports it.

The stage adds no rebuild item of its own: G3's build recomputes and rewrites its arrays and record even when the
record exists.

**Disk:** about 0.05 GB, G5's figure for zspx's stage (docs/G5_COMBINED_GNN.md, section 4); zspxt's fits are the same
size.

**ETA, an estimate.** No run time of A9's or A9t's fits is filed in the repo. G5 recorded that G3's and G4's fits took
47 to 54 minutes each on the card and their reads about 5 minutes, and expected about 1 hour 15 minutes for zspx's two
fits at once (docs/G5_COMBINED_GNN.md, section 4). zspxt costs what zspx costs: the bound adds a few operations per
row, and every edge is read, as in zspx. So, as an estimate, about 1 hour 15 minutes to 1 hour 30 minutes from the
trains' start to the verdict, once the card has room for two 0.42 shares. When another workload held part of the card
on 19 October, runs beside it went 7 to 60 times slower (docs/WHERE_WE_STAND.md).

The encoder stays frozen. webqsp never trains. Test splits are never read. No LLM or transformer.

## 7. Results

Not run yet.
