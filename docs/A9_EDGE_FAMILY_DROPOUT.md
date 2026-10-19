# A9: edge-family dropout on the adopted GNN

Declared 17 October 2026, before any number. The ninth reopening of docs/VERDICT_AUDIT_2026_10_17.md (section 2,
row 9), under docs/SCREENS.md section 3b. Code: `outputs/mp_unified/za9fd.py` (the arm, the screen and the call).
Filed 19 October: zspxd NOT_ADOPTED (section 7).

## 1. Why this, now

**Where the idea came from.** The MP-unified track (early October; docs/STATUS_2026_10_05.md) trained one-graph
message passing with edge-family dropout: in training, each question drops each link family it has with probability
0.25 (qd_gnn13's fd25 rule). It was the first one-graph message-passing arm with no unseen graph below the edge-free
twin on R@5. Without it, message passing over one graph hurt zero-shot reads.

**It never reached the GNN on its own.** zsp, zspk and G3 to G5 all train without it. G2 (15 October) carried it, but
only together with three hidden-state layers and a bounded readout (0 GAIN, 0 LOSS). That verdict closed G2's form,
not the dropout. Under section 3b the idea stays open until it is tried alone on the GNN.

**The base is zspx.** The audit row reads "zspk (zspx if G5 passes)". G5's zspx passed (ADOPT_FOR_FULL_RUN; 2 GAIN,
both on L-hotpotqa: musique +0.0147 and webqsp zero-shot +0.0235; 0 LOSS). Its three steps all read the link edges,
so the dropout touches every place where zspx passes messages.

**What the dropout is for.** The dropout keeps the model from leaning on one family. A model trained on four datasets'
graphs then meets graphs whose family mix differs (webqsp's KB, a held-out passage graph). If zspx's gains lean on the
training graphs' family mix, the zero-shot reads show it.

## 2. The form

zspxd is zspx (outputs/mp_unified/zg5mix.py) with one change, in training only.

**The change.** Before zspx's forward, each question's link edges lose the families the dropout draws:
- each question drops each family it has where its draw u < 0.25;
- when every family it has would go, it keeps its structural family (0), or without one its lowest;
- the rule is zg2.family_keep's, the same code G2 ran;
- G3's per-edge bridge residual s goes with its edge, and the per-row c is unchanged.

All three steps of zspx then read the same kept edges in that forward: its step with the gradient kept, G3's residual
message and G4's second step.

**At read,** every edge is kept and no draw is made. zspxd's read is zspx's forward on its own weights.

**Draws.** One (B, 3) uniform draw per forward, from the module's own generator (seed + 9901). No draw of zspx's
moves: the scorer's dropout, batches and initial weights are zspx's. With no edge dropped (FD 0), zspxd's fit is
zspx's bit for bit.

**What does not change.** No parameter is added: zspxd has zspx's three step heads, 385 parameters each. No new graph, column,
encoder or text. Settings, seeds, batches, KT input, loss and loop are all zspx's (that is, zspk's).

**GNN only.** The MLP has no edges, so it has nothing to drop.

## 3. The arm

| arm | the model | the carve |
| --- | --- | --- |
| zspxd | zspx's, its steps reading the dropout's kept edges in training | zspx's (G3's: zspk's with c and s beside) |

## 4. The fits and the call

- **Fits:** one card fit on L-musique and one on L-hotpotqa (`scr-a9-zspxd`, `-hp`), on P1c2's roots with P1c3's KT
  input. Card fit settings: variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128 (zspx's).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against zspx's card fit of the same split (scr-g5-zspx, scr-g5-zspx-hp), which decides.
  Step 1's fit is reported beside.
- **The re-call's base screens:** zspx's card fits compared with step 1's of the same split
  (outputs/a9/base-zspx-<split>), as L1 did for zspk.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md). GAIN, LOSS and
  WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

- **ADOPT_FOR_FULL_RUN:** zspxd's re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A9's re-call of zspxd, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- webqsp and metaqa, both splits (the KB graphs: webqsp is the zero-shot KB);
- musique in domain (L-hotpotqa): Claim 1's read;
- hit@1 and FC@5 per read.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | zspxd fitted again on P1g-b's pools, then its full run beside zspx's (its own declared file) |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A9r, a lighter rate: the same rule at FD 0.1, so each question keeps all its families on most steps (one declared file) |
| NOT_ADOPTED, with no GAIN | A9t, the MP-unified track's whole form: the dropout, with zspx's three increments bounded together, s0 + 0.5 tanh((s3 − s0) / 0.5) (qd_gnn13's tb5, G2's bound) (one declared file) |
| INCOMPLETE | the missing re-call is run; nothing else changes |

No model change is adopted on today's pools (the user's rule of 13 October).

## 6. How it runs

**Self-test, passed before this commit.** `python outputs/mp_unified/za9fd.py --selftest` (8 s) checks, on a toy
carve with G3's residuals:
- **The arm.** It is zspx's with its own generator, on G3's carve; zspx's and zspk's arms are unchanged; the rate is
  0.25 over three families.
- **The rule.** family_keep matches a plain-Python rule on draws that drop some families, every family and none. With
  every family drawn, each question keeps exactly one family: 0 where it has it, else its lowest.
- **The kept edges.** They carry their s; every other key, and c, is unchanged; the input is not modified.
- **Bit-for-bit cases.**
  - zspxd's state equals zspx's from the same seed.
  - At read, zspxd equals zspx with every edge, with no draw made.
  - In training, zspxd equals zspx over the kept edges, its draws from its own generator.
  - With FD 0, and without links, zspxd is zspx.
- **Toy fits through lean_gpu's loop.**
  - A repeat is bit for bit, and the dropout moves the fit off zspx's.
  - With FD 0, and without links, the fit is zspx's bit for bit.
- **The rest.** The train and read refusals; relz's pair and re-call stamped A9's; the verdict's outcomes and next
  forms.

**Host items** (`a9-*`), after A8's verdict (11 items):
- the two base compares (zspx's card fits against step 1's), no dependency;
- 2 trains and 2 reads, GPU sized as G5's from zspx's runs: `--frac 0.40` (share 0.42); memory: trains 6 GB
  (-hp 15 GB), reads 6 GB;
- 2 compares, the pair, the re-call (after the base compares) and the verdict.

## 7. Results

Filed 19 October 2026 from the host's records (`outputs/a9/screen-verdict.json`, written 2026-10-18T00:01:12Z; the
re-calls `outputs/screen/scr-a9-zspxd-pair-recall.json`, host paths made relative). **zspxd NOT_ADOPTED:** no GAIN and
no LOSS after the seed-null re-call.

- **zspxd: 0 GAIN, 0 LOSS.** On L-hotpotqa, hotpotqa zero-shot falls 0.0075 [-0.0102, -0.0046] and musique in domain
  (Claim 1's read) 0.0059 [-0.0108, -0.0012], both WITHIN; every other read is within 0.004.
- **Beside, the KB graphs:** webqsp zero-shot -0.0034 (L-musique) and -0.0037 (L-hotpotqa); metaqa in domain -0.0019 and
  +0.0007, all WITHIN.

Dropping whole edge families in training moves no read past the seed null on zspx, and costs Claim 1's musique read a
little. This closes only this form (the dropout alone, the increments unbounded). The next form, fixed in section 5 and
recorded in the verdict, is **A9t**, the MP-unified track's whole form: the dropout with zspx's three increments bounded
together, s0 + 0.5 tanh((s3 - s0) / 0.5) (qd_gnn13's tb5, G2's bound); one declared file.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's; a call is shown with its
filed call where the re-call changed it; base R@5 shown as zspx's):

| split | dataset | read | base R@5 | zspxd: delta [95% CI], call |
| --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8030 | -0.0019 [-0.0029, -0.0011], WITHIN |
| L-musique | squad | in-domain | 0.9092 | +0.0026 [+0.0008, +0.0045], WITHIN |
| L-musique | musique | zero-shot | 0.5073 | +0.0024 [-0.0036, +0.0079], WITHIN |
| L-musique | hotpotqa | in-domain | 0.9074 | -0.0015 [-0.0036, +0.0006], WITHIN |
| L-musique | 2wiki | in-domain | 0.9351 | -0.0005 [-0.0018, +0.0009], WITHIN |
| L-musique | webqsp | zero-shot | 0.3204 | -0.0034 [-0.0102, +0.0037], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8039 | +0.0007 [-0.0002, +0.0015], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9129 | -0.0008 [-0.0024, +0.0010], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5901 | -0.0059 [-0.0108, -0.0012], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8547 | -0.0075 [-0.0102, -0.0046], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9312 | +0.0026 [+0.0012, +0.0041], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2914 | -0.0037 [-0.0121, +0.0043], WITHIN |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a9-zspxd.json` and the others, `-hp` for L-hotpotqa):

| split | dataset | zspxd FC@5 | zspxd hit@1 |
| --- | --- | ---: | ---: |
| L-musique | metaqa | -0.0016 | -0.0097 |
| L-musique | squad | +0.0026 | -0.0002 |
| L-musique | musique | -0.0046 | +0.0223 |
| L-musique | hotpotqa | -0.0018 | -0.0035 |
| L-musique | 2wiki | -0.0012 | +0.0046 |
| L-musique | webqsp | -0.0020 | -0.0027 |
| L-hotpotqa | metaqa | +0.0004 | -0.0020 |
| L-hotpotqa | squad | -0.0008 | -0.0003 |
| L-hotpotqa | musique | -0.0153 | -0.0041 |
| L-hotpotqa | hotpotqa | -0.0149 | +0.0000 |
| L-hotpotqa | 2wiki | +0.0064 | +0.0001 |
| L-hotpotqa | webqsp | +0.0013 | +0.0013 |
