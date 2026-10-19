# A9t: the MP-unified track's whole form on the adopted GNN

Declared 19 October 2026, before any number. The next form A9's verdict named for no GAIN
(docs/A9_EDGE_FAMILY_DROPOUT.md, section 5; outputs/a9/screen-verdict.json), under docs/SCREENS.md section 3b. Code:
`outputs/mp_unified/za9t.py` (the arm, the screen and the call). Filed 20 October: NOT_ADOPTED with no GAIN and one LOSS
(musique in domain on L-hotpotqa); the next form is A9b (section 5).

## 1. Why this, now

**What A9 found.** zspxd, zspx trained with the track's edge-family dropout, was NOT_ADOPTED on 18 October: 0 GAIN,
0 LOSS. Its re-call under the seed null changed no call; all twelve reads are WITHIN. Seven reads did not move: their
95% intervals hold zero. Five moved, all inside their floors:

| read | zspx R@5 | zspxd R@5 | difference [95% CI] | floor |
| --- | ---: | ---: | --- | ---: |
| L-musique metaqa (in domain) | 0.8030 | 0.8011 | −0.0019 [−0.0029, −0.0011] | 0.0075 |
| L-musique squad (in domain) | 0.9092 | 0.9118 | +0.0026 [+0.0008, +0.0045] | 0.0075 |
| L-hotpotqa musique (in domain) | 0.5901 | 0.5842 | −0.0059 [−0.0108, −0.0012] | 0.0075 |
| L-hotpotqa hotpotqa (zero-shot) | 0.8547 | 0.8472 | −0.0075 [−0.0102, −0.0046] | 0.0200 |
| L-hotpotqa 2wiki (in domain) | 0.9312 | 0.9339 | +0.0026 [+0.0012, +0.0041] | 0.0075 |

The zero-shot KB read did not move either way: webqsp −0.0034 [−0.0102, +0.0037] from L-musique and −0.0037
[−0.0121, +0.0043] from L-hotpotqa. So the dropout alone did not protect a zero-shot read, and two of its five moves
were down on L-hotpotqa, the split where G5's zspx had its two GAINs (musique +0.0147, webqsp zero-shot +0.0235).

**What the track ran.** qd_gnn13 added the dropout (`-fd25`) to arms that already bounded what message passing adds
to the score before it (`-tb5`: s = z + 0.5 tanh((s_mp − z) / 0.5), qd_gnn7's `-tb` rule; for example
`QD-TXT-L2-dr25e10-tb5-ed50-at-rg4-fd25`). Unbounded message passing on one graph hurt zero-shot reads there. G2
(15 October) carried both, but over its own stack's readout on zspk. Neither the bound nor the pair has acted on
zspx's three steps.

**Why that closes only its form.** A9's verdict closed the dropout alone. Under section 3b the idea stays open until
its whole form has reached the adopted GNN. A9's section 5 named that form for no GAIN: this stage.

## 2. The form

zspxdt is zspxd (outputs/mp_unified/za9fd.py) with one change: the score after zspx's three steps is bounded around the
score before the first.

**Where s0 and s3 are taken** (zspx's forward, `zg5mix.ZSPX.forward`, lines 93 to 101):
- **s0** is line 94: `s = super(ZP.ZProp, self).forward(feats, keep, nq, B, base_z)`, the score of zspk's scorer and
  KT head, before any step. It reads no edge.
- s1 is line 98: zspk's step, its gradient kept (zspkg).
- s2 is line 100: G3's residual message (on a carve with G3's residuals).
- **s3** is line 101: the score zspx's forward returns, after G4's second step.

**The change.** zspxdt returns

    out = s0 + 0.5 tanh((s3 − s0) / 0.5)

in place of s3. The three increments are bounded together, not each. Each step still reads the unbounded score before
it (s0, s1, s2), as in zspx; the bound acts only on what the forward returns. Without links (lines 95 and 96) the
forward returns s0, and there is nothing to bound.

**The width.** 0.5, qd_gnn13's tb5 (qd_gnn7's `-tb<T>` rule, t = T / 10), the width G2 used (zg2.TB). A move s3 − s0
of 0.1 becomes 0.0987; a move of 1.0 becomes 0.482; no move passes 0.5. The track's EM arms also ran a 0.2 width
(qd_gnn14's tb2); that width is not this form.

**The gradient.** It is kept through s0 and s3; nothing is detached. With x = (s3 − s0) / 0.5,
∂out/∂θ = (1 − sech² x) ∂s0/∂θ + sech² x ∂s3/∂θ. The scorer keeps its full path; the steps' increments get sech² x of
theirs, which is 1 where the move is small.

**When.** In training and at read.

**The dropout is A9's, unchanged.** In training each question drops each link family it has where its draw u < 0.25
(zg2.family_keep; the structural family, or else the lowest, is kept when all would go). G3's per-edge s goes with its
edge. All three steps read the kept edges; s0 reads none. The draws come from zspxd's own generator (seed + 9901). At
read every edge is kept and no draw is made.

**Code.** `za9t.BoundedSteps.forward` restates zg5mix's lines 94 to 101 line for line (zg5mix is frozen), keeps s0,
and returns the bound. `ZSPXDT` is zspxd over the bounded steps (MRO: ZSPXD, BoundedSteps, ZSPX): zspxd's forward drops
the families in training and hands the kept edges to the bounded steps.

**What does not change.** No parameter is added: zspxdt has zspx's three step heads, 385 parameters each. No new graph,
column, encoder or text. Settings, seeds, batches, KT input, loss and loop are zspx's (that is, zspk's).

**Bit-for-bit anchors.** At the start every step head's last layer is zero, so s3 = s0 and zspxdt is zspx (and
zspxd). With the bound off, zspxdt is zspxd. With the bound off and FD 0, it is zspx.

**GNN only.** The MLP takes no message, so it has nothing to bound or drop.

## 3. The arm

| arm | the model | the carve |
| --- | --- | --- |
| zspxdt | zspxd's (zspx with A9's dropout in training), its three increments bounded together at 0.5 | zspx's (G3's: zspk's with c and s beside) |

## 4. The fits and the call

- **Fits:** one card fit on L-musique and one on L-hotpotqa (`scr-a9t-zspxdt`, `-hp`), on P1c2's roots with P1c3's KT
  input. Card fit settings: variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128 (zspx's).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against zspx's card fit of the same split (scr-g5-zspx, scr-g5-zspx-hp), which decides.
  Step 1's fit is reported beside.
- **The re-call's base screens:** A9's, zspx's card fits compared with step 1's of the same split
  (outputs/a9/base-zspx-<split>).
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md). GAIN, LOSS and
  WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

- **ADOPT_FOR_FULL_RUN:** zspxdt's re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A9t's re-call of zspxdt, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- webqsp and metaqa, both splits (the KB graphs: webqsp is the zero-shot KB);
- musique in domain (L-hotpotqa): Claim 1's read;
- hit@1 and FC@5 per read;
- zspxd's difference per read from A9's own re-call (the dropout alone), so the bound's part can be read beside it.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | zspxdt fitted again on P1g-b's pools, then its full run beside zspx's (its own declared file) |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A9tq, the learned back-off: the bound's width per question, 0.5 · 2σ(a · m_q + b), m_q the shares of the question's link edges in the three families; a and b start at zero, so A9tq starts as A9t; the dropout kept (one declared file) |
| NOT_ADOPTED, with no GAIN | A9b, the bound alone: zspx's three increments bounded together at 0.5, with no dropout, the one part of the track's form not yet on zspx (one declared file). If A9b finds no GAIN either, the track's form closes on the GNN: its dropout (A9), its bound (A9b) and both together (A9t) will all have reached the adopted GNN, and the MLP takes no message |
| INCOMPLETE | the missing re-call is run; nothing else changes |

No model change is adopted on today's pools (the user's rule of 13 October).

## 6. How it runs

**Self-test, passed before this commit.** `python outputs/mp_unified/za9t.py --selftest` (12 s) checks, on a toy carve
with G3's residuals:
- **The arm.** zspxd's model over the bounded steps, on G3's carve; zspxd's, zspx's and zspk's arms are unchanged; the
  width is G2's 0.5, the rate A9's 0.25 over three families; A9's base fits and base screens are the re-call's.
- **The bound by hand.** s0 + 0.5 tanh((s3 − s0) / 0.5) against math.tanh on moves from −3 to 3, and its gradient:
  1 − tanh² for s3, tanh² for s0.
- **Bit-for-bit cases.**
  - zspxdt's state equals zspx's and zspxd's from the same seed, and its generator is zspxd's.
  - At the start, zspxdt is zspx, and both are zspx's scorer (s3 = s0).
  - With all three step heads moved, small and large: with the bound off zspxdt is zspx; with it on it is the bound of
    s0 (zspx's scorer) and s3 (zspx's score). Every move keeps its sign, shrinks toward s0 and stays within 0.5.
    With small moves (all below 0.05) zspxdt stays within 1% of the largest move from zspx; moves past 1 saturate
    (above 0.482).
  - The three increments are bounded together: the result differs from bounding each step's increment alone.
  - In training, zspxdt is the bound over zspx on the kept edges, its draws zspxd's; with the bound off it is zspxd;
    with FD 0 it is the bound over zspx on every edge; at read no draw is made.
- **The gradient.** Every parameter's gradient is (1 − sech²) times the scorer's plus sech² times zspx's, and the
  three heads' gradients differ from zspx's.
- **Toy fits through lean_gpu's loop.** A repeat is bit for bit; the bound moves the fit off zspxd's; with the bound
  off the fit is zspxd's bit for bit, with FD 0 too it is zspx's, and without links it is zspx's.
- **The rest.** The train and read refusals; relz's pair and re-call stamped A9t's; the verdict's outcomes and next
  forms, with zspxd's reads beside from A9's re-call.

**Commands.**

    python outputs/mp_unified/za9t.py train --split L-musique --name scr-a9t-zspxdt --arm zspxdt --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za9t.py train --split L-hotpotqa --name scr-a9t-zspxdt-hp --arm zspxdt --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za9t.py read --name scr-a9t-zspxdt --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za9t.py read --name scr-a9t-zspxdt-hp --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za9t.py compare --new outputs/screen/fits/scr-a9t-zspxdt --base outputs/screen/fits/scr-g5-zspx,outputs/step1/fits/L-musique --out outputs/screen/scr-a9t-zspxdt
    python outputs/mp_unified/za9t.py compare --new outputs/screen/fits/scr-a9t-zspxdt-hp --base outputs/screen/fits/scr-g5-zspx-hp,outputs/step1/fits/L-hotpotqa --out outputs/screen/scr-a9t-zspxdt-hp
    python outputs/mp_unified/za9t.py pair --screens outputs/screen/scr-a9t-zspxdt.json,outputs/screen/scr-a9t-zspxdt-hp.json --out outputs/screen/scr-a9t-zspxdt-pair
    python outputs/mp_unified/za9t.py recall --null outputs/screen/scr-null-s1.json,outputs/screen/scr-null-s2.json,outputs/screen/scr-null-s1-hp.json,outputs/screen/scr-null-s2-hp.json --pair outputs/screen/scr-a9t-zspxdt-pair.json --out outputs/screen/scr-a9t-zspxdt-pair-recall
    python outputs/mp_unified/za9t.py verdict

**Host items** (`a9t-*`, 9 items):
- 2 trains and 2 reads, sized as A9's from zspx's runs: `--frac 0.40` (share 0.42); memory: trains 6 GB (-hp 15 GB),
  reads 6 GB;
- the trains wait on the rebuild of G3's residual arrays (the `n19-res-*` items) for their split's fit carves (11 for
  L-musique, 16 for L-hotpotqa), the reads on the six s1eval carves' (22);
- 2 compares, the pair, the re-call (after A9's base compares, `a9-base-zspx` and `-hp`) and the verdict.

The stage adds no rebuild item of its own: G3's build recomputes and rewrites its arrays and record even when the
record exists.

## 7. Results

Filed 20 October 2026 from the host's records (`outputs/a9t/screen-verdict.json`, written 2026-10-19T19:44:27Z; the
re-calls `outputs/screen/scr-a9t-zspxdt-pair-recall.json`, host paths made relative). **NOT_ADOPTED, with no GAIN and
one LOSS:** with zspx's three increments bounded together, A9's edge-family dropout moves no read up past the seed null,
and musique in domain on L-hotpotqa falls past its floor.

- **zspxdt: 0 GAIN, 1 LOSS.** The LOSS is musique in domain on L-hotpotqa: -0.0125 [-0.0177, -0.0074] against its 0.0075
  floor, with FC@5 -0.0219. webqsp zero-shot on L-hotpotqa falls -0.0208 [-0.0287, -0.0131], a LOSS by its pair's
  interval that is WITHIN under its 0.0233 floor. Five more intervals exclude 0, all inside their floors: on L-musique,
  metaqa -0.0019 [-0.0030, -0.0009] and squad +0.0035 [+0.0015, +0.0054]; on L-hotpotqa, metaqa -0.0023 [-0.0032,
  -0.0014], hotpotqa zero-shot -0.0035 [-0.0064, -0.0006] and 2wiki -0.0047 [-0.0062, -0.0032].
- **Beside: with the bound added, A9's falls on L-hotpotqa grew.** With the dropout alone (A9), zspxd moved musique on
  L-hotpotqa -0.0059 and webqsp zero-shot there -0.0037; with the increments bounded too, the moves are -0.0125 and
  -0.0208. musique zero-shot on L-musique turns from +0.0024 to -0.0012.

Not adopted. This closes only this form: A9's edge-family dropout with zspx's three increments bounded together, in
training and at read (docs/SCREENS.md section 3b). The next form, fixed in section 5 and recorded in the verdict, is
A9b: the bound alone, zspx's three increments bounded together at 0.5 with no dropout, the one part of the MP-unified
track's form not yet on zspx, in its own declared file. If A9b finds no GAIN either, the track's form closes on the GNN:
its dropout (A9), its bound (A9b) and both together (A9t) will all have reached the adopted GNN, and the MLP takes no
message.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's; a call is shown with its
filed call where the re-call changed it; base R@5 shown as zspx's):

| split | dataset | read | base R@5 | zspxdt: delta [95% CI], call |
| --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8030 | -0.0019 [-0.0030, -0.0009], WITHIN |
| L-musique | squad | in-domain | 0.9092 | +0.0035 [+0.0015, +0.0054], WITHIN |
| L-musique | musique | zero-shot | 0.5073 | -0.0012 [-0.0077, +0.0049], WITHIN |
| L-musique | hotpotqa | in-domain | 0.9074 | -0.0002 [-0.0026, +0.0020], WITHIN |
| L-musique | 2wiki | in-domain | 0.9351 | -0.0010 [-0.0025, +0.0004], WITHIN |
| L-musique | webqsp | zero-shot | 0.3204 | -0.0012 [-0.0095, +0.0065], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8039 | -0.0023 [-0.0032, -0.0014], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9129 | +0.0007 [-0.0012, +0.0025], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5901 | -0.0125 [-0.0177, -0.0074], **LOSS** |
| L-hotpotqa | hotpotqa | zero-shot | 0.8547 | -0.0035 [-0.0064, -0.0006], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9312 | -0.0047 [-0.0062, -0.0032], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2914 | -0.0208 [-0.0287, -0.0131], WITHIN (filed LOSS) |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a9t-zspxdt.json` and the others, `-hp` for L-hotpotqa); and A9's R@5 deltas with the dropout alone
(zspxd, its own re-call), on the same read:

| split | dataset | zspxdt FC@5 | zspxdt hit@1 | zspxd R@5 (A9) |
| --- | --- | ---: | ---: | ---: |
| L-musique | metaqa | -0.0014 | -0.0072 | -0.0019 |
| L-musique | squad | +0.0035 | +0.0045 | +0.0026 |
| L-musique | musique | -0.0124 | -0.0074 | +0.0024 |
| L-musique | hotpotqa | +0.0005 | -0.0080 | -0.0015 |
| L-musique | 2wiki | -0.0015 | +0.0056 | -0.0005 |
| L-musique | webqsp | +0.0007 | -0.0047 | -0.0034 |
| L-hotpotqa | metaqa | -0.0036 | -0.0132 | +0.0007 |
| L-hotpotqa | squad | +0.0007 | +0.0017 | -0.0008 |
| L-hotpotqa | musique | -0.0219 | -0.0116 | -0.0059 |
| L-hotpotqa | hotpotqa | -0.0063 | -0.0084 | -0.0075 |
| L-hotpotqa | 2wiki | -0.0177 | -0.0024 | +0.0026 |
| L-hotpotqa | webqsp | -0.0133 | -0.0120 | -0.0037 |
