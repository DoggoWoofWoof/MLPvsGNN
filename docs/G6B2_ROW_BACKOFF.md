# G6b2: zg6b with a per-row back-off to its twin, on Kaggle's T4

Declared 27 October 2026. Code: `outputs/mp_unified/zg6b2.py`. Its fits run in KG's sessions (`zkg_session.py`) on KG's
device. No G6b2 fit has run. Nothing in this file reads a G6b2 number.

## 1. Why this, now

- **G6b's call fixes this form.** G6b was filed NOT_ADOPTED on 27 October (docs/G6B_GATED_BACKOFF.md, section 9): 3 GAIN
  and 3 LOSS against KG's zspk, re-called under KG's zspk null. Before any number, G6b's section 6 fixed the next form
  for that outcome: "G6b2, a per-row back-off: the gate also reads the row's own log1p in-degree per family over the
  edges that reach the layers, so a row the edges barely reach can lean on the twin (one declared file)". This is that
  file.
- **What G6b's gate learned.** It is close to one fixed mix per split. Its mean weight on the edge pass is 0.34 to 0.40
  on L-musique and 0.29 to 0.37 on L-hotpotqa. Within a dataset, its 10th to 90th percentiles span only 0.008 to 0.053,
  with squad the exception. The question's family shares barely separate one question from another. The back-off
  recovered about half of zg6's loss on L-musique's webqsp read, and none on L-hotpotqa's zero-shot reads.
- **Why the row.** Within one question, the edges reach some rows and not others, and they reach them in different
  numbers.
  - A row that no edge reaches already scores as the twin, in zg6b as in zg6b2 (section 2).
  - Every reached row of a question gets G6b's one weight, whether one edge reaches it or a hundred.
  - zg6b2 lets the weight move with how many edges of each family reach the row. The fit decides the direction: trust
    the edges where many arrive and lean on the twin where few do, or the reverse.
- **What it reads.** It reads no dataset, split, gold or row label. The in-degrees count the question's own pool edges
  (structural, NER and kNN), the same edges the layers read. The form stays G6's: a free GNN with its own encoder,
  messages between hidden states and an unbounded readout.
- **The device is KG's T4**, as for G6b. The lab host is offline, workspace 3 renews on 1 November, and KG's fits are
  the T4's bases and nulls (docs/KG_KAGGLE_T4_LINE.md, section 10).
- **The risks, stated before any number.**
  - The gate learns only from the training splits' questions, which are in-domain, as G6b's did.
  - If a row's in-degree does not separate, on the training graphs, the rows where the edges help from those where
    they hurt, c stays near 0 and zg6b2 is zg6b.
  - The zero-shot datasets' graphs can have other in-degree profiles than the training splits'. The gate is linear in
    log1p in-degree, so there it extrapolates. Section 5's gate statistics show each dataset's in-degrees and the gate
    on its reached rows.

## 2. The model

zg6b2 is zg6b's network (G6b, section 2), unchanged:
- zg6's encoder, boundary, context, three layers and readout, its edge-family dropout in training (with the same draws
  from the module's own generator), and its training (rmatch.py's train on P1c2's card, as zspk's, zg6's and zg6b's
  fits);
- scored twice from one encoder pass and one context: **s_e** over the question's pool edges that reach the layers
  (zg6's score), and **s_t** with no edges (the twin's score).

What is new: the gate moves from the question to the row.
- **The row's in-degree.** For row r, deg_r holds the number of edges of each family that reach the layers and point at
  r. An edge u -> v counts at its target v. In training, the edges the family dropout drops are not counted. A row that
  no edge reaches has deg_r = 0. This is the count zg6's layers already divide their per-family means by, before their
  clamp at 1.
- **The gate.** g_r = sigmoid(a · sh_q + c · log1p(deg_r) + b), where:
  - sh_q is zg6b's family shares of the row's question q, unchanged;
  - a (three weights) and b are zg6b's gate;
  - c is three new weights, one for each family.

  a, b and c start at 0, so g = 0.5 on every row at the start.
- **The score**, on row r: g_r · s_e,r + (1 - g_r) · s_t,r. At the start this is zg6b's score, bit for bit.
- **A row no edge reaches** has s_e = s_t bit for bit. The question's context enters a layer only through the attention
  logits of the edges into a row, so such a row's state is the same in both passes at every layer. Its score is the
  twin's whatever g_r is, and it gives the gate no gradient. The gate learns only from reached rows.
- **Parameters.** 121,809 of its own at hidden 128: zg6b's 121,806 and c's 3. From seed 0, every weight starts where
  zg6b's does, c starts at 0, and the dropout's generator starts where zg6b's does.

The gate reads only the edges' family shares and the row's counts over those same edges. It does not read the
dataset, the split or any gold.

## 3. The fits and the device

Two fits on U2d's base build, seed 0:

| G6b2 fit | split | its bases, all on the same T4 |
| --- | --- | --- |
| `scr-g6b2-zg6b2` | L-musique | KG's `scr-kg-base-zspk` (decides); KG's `scr-kg-base-zg6` and G6b's `scr-g6b-zg6b` (beside) |
| `scr-g6b2-zg6b2-hp` | L-hotpotqa | KG's `scr-kg-base-zspk-hp` (decides); KG's `scr-kg-base-zg6-hp` and G6b's `scr-g6b-zg6b-hp` (beside) |

- **The same path as KG's and G6b's.** zg6b2.py trains and reads through zu2d.py's train and read bodies, as zkg.py
  runs them:
  - U2d's base build's roots and gate, P1c2's card, and seed 0;
  - reads of p@swa on the six s1eval carves with `--rows-cap 65536` (whole questions, so the scores do not change).

  Its stamp records the device, the build, the pools, the KT, rmatch, zrc and zlink records, the model's settings, and
  its bases.
- **The device is KG's.** Train and read run only in a Kaggle kernel on a T4 from KG's image
  (kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3), checked before any data is read. A read must run on its fit's
  device.
- **Before G6b2 is called:**
  - KG's base verdict (outputs/kg/base-verdict.json) must be BASE_READY.
  - G6b's verdict (outputs/g6b/screen-verdict.json) must be NOT_ADOPTED with G6b2 as its next form.
  - G6b2's two fits must be on KG's device.
- **Memory.** zg6b's peaks on the T4 were 4.89 GB in training and 5.24 GB reading on L-musique, and 10.29 GB on
  L-hotpotqa; the T4 has 16.1 GB. The in-degrees add one N x 3 tensor. Each fit's g6b2_timing.json records its peaks.

## 4. The compares, the null and the rule

- **Two compares**, `scr-g6b2-zg6b2-vs-zspk` and `-hp`. Each is lean_screen's compare (p@swa on s1eval) of zg6b2's fit
  against three bases:
  - KG's zspk, which comes first and decides;
  - KG's zg6 and G6b's zg6b, beside.

  All four fits must be on one device. Each compare is stamped with G6b2's record.
- **The call.**
  1. screen_pair over the two compares against zspk: `outputs/g6b2/g6b2-zg6b2-pair`.
  2. A re-call under KG's zspk seed null: `outputs/g6b2/g6b2-zg6b2-pair-recall`. KG's floors are screen_recall's,
     max(0.0075, 2·sqrt((D1² + D2²)/2)), from KG's seed-1 and seed-2 zspk fits.
- **The rule** is G6's (G6, section 4), on the twelve reads:
  - **ADOPT_FOR_FULL_RUN:** no LOSS on any read and at least one GAIN.
  - **NOT_ADOPTED:** otherwise.
  - **INCOMPLETE:** any of these:
    - KG's base verdict is not BASE_READY;
    - G6b's verdict is not NOT_ADOPTED with G6b2 next;
    - a fit or compare is missing, or is not G6b2's;
    - the fits are not on KG's device;
    - a compare's zg6 or zg6b column is not that fit's read as filed (section 5);
    - a re-call does not hold the twelve reads.

    The verdict names what is missing.

## 5. Beside, deciding nothing

- **zg6b2 against zg6**, the compares' second base, through the same pair and re-call
  (`outputs/g6b2/g6b2-zg6b2-vs-zg6-pair` and `-recall`).
  - KG fits no zg6 seed null, so the floors are zspk's, as a proxy.
  - zg6's seed-0 R@5 comes from KG's own `scr-kg-zg6-vs-zspk` compares. Each G6b2 compare's zg6 column must equal it.
- **zg6b2 against zg6b**, the third base, the same way (`outputs/g6b2/g6b2-zg6b2-vs-zg6b-pair` and `-recall`).
  - The floors are zspk's.
  - zg6b's seed-0 R@5 comes from G6b's filed `scr-g6b-zg6b-vs-zspk` compares (their zg6b column). Each G6b2 compare's
    zg6b column must equal it.
  - This read says whether moving the gate to the row changes anything over G6b.
- **R@5 per read on the T4** for rrf, zspk, zg6, zg6b, zg6b2 and zrckp (KG's seed-0 fit, the MLP). The MLP share,
  (zrckp - rrf) / (zg6b2 - rrf), is shown where zg6b2's gain over rrf exceeds 0.002.
- **The gate** of each fit's p@swa:
  - a, b and c;
  - its value on a row of a question of one family alone: unreached (the question part alone), and reached by 1, 3,
    10, 30 and 100 edges of that family; and on a row of a question without edges;
  - over each dataset's s1eval questions: the question part sigmoid(a · sh + b) (its mean, and its 10th, 50th and 90th
    percentiles) and the share of questions without edges;
  - over the rows the edges reach: their share of the carve's rows, the row gate (its mean and percentiles), and their
    mean log1p in-degree per family.

  The kernel computes these from zlink's arrays, each checked against its part's record, and writes g6b2_gate.json. A
  row's in-degree there counts the read's edges for each question and target row, as the model does at read. The
  laptop holds only the arrays' records.
- **Seconds and memory** for each fit, beside G6b's zg6b and KG's zg6 of the same split on the same T4.

## 6. What each outcome leads to (fixed now)

| zg6b2's re-call under KG's zspk null | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | zg6b2 fitted again on U2s's carried pools, then its full run on all six splits beside zspk's, then Claim 1 (P3c's intervals) and Claim 3 (cold latency, the C stages' timing) measured again; each its own declared file |
| NOT_ADOPTED, with or without a GAIN | G6r (G6, section 5): relation-aware messages from text, the relation embeddings (`outputs/m3b/relations/metaqa_rel_embeddings.npy`, `webqsp_rel_embeddings.npy`) through one shared low-rank map in place of the per-family maps on the KB datasets, the encoder started from zspk's trained encoder, still unbounded with its own readout (one declared file) |
| INCOMPLETE | what is missing is supplied (KG's base set, G6b's verdict, a fit, a compare or a re-call); nothing else changes |

- **Why G6r follows a GAIN and a LOSS too.** With G6b2, the back-off has had both of its forms, per question and per
  row. If the mix still trades zero-shot reads for in-domain ones, the next change is to the messages, not the mix.
- **Why U2s's pools.** G6b named P1g-b's pools for its refit. P1g-b was not adopted, and U2s (the walk-mass budget, one
  construction for all six datasets) is the pools stage the program now carries, as U2n's section 6 names it.

No verdict here closes the free GNN (docs/SCREENS.md, section 3b). No model change is adopted on today's pools (the
user's rule of 13 October).

## 7. How it runs

- **In KG's sessions.** zkg_session.py runs each queued fit through its stage's script, each in its own process:
  - `zkg.py fit` for scr-kg- names;
  - `zg6b.py fit` for scr-g6b- names;
  - `zg6b2.py fit` for scr-g6b2- names, added in this commit (KG, section 4, amended).

  It refuses a name that no stage there fits, or a fit queued twice, before reading any data.
- **zkg.py is not changed.** Its take stops on a scr-g6b2- folder, so it is never run on a G6b2 session's output.
  `zg6b2.py take` takes only scr-g6b2- folders, and zg6b.py's take only scr-g6b- ones.
- **The session.** One session of kernel mpr-kg after U2n's session 3, at this commit:
  - GPU 0: `scr-g6b2-zg6b2`;
  - GPU 1: `scr-g6b2-zg6b2-hp`.

  Where a fit runs decides nothing.
- **Time.** zg6b's fits took 3,134 s (L-musique) and 5,151 s (L-hotpotqa) on the T4, train and read together. zg6b2
  adds one count per edge and three weights, so the session should take about 1.5 hours plus the datasets' check, and
  about 2.3 T4-hours in all. It costs nothing (KG, section 8).
- **Commands.** In the kernel, the stub runs the session, and zkg_session runs
  `zg6b2.py fit --name N --rows-cap 65536` for each G6b2 name. On the laptop, after the session:

      python outputs/mp_unified/zg6b2.py take --from <the kernel's output folder>
      python outputs/mp_unified/zg6b2.py compare-all
      python outputs/mp_unified/zg6b2.py verdict

- **What is committed.**
  - Committed: G6b2's compares (outputs/screen/scr-g6b2-*) and outputs/g6b2: the pairs, the re-calls and the verdict,
    which carries the gate statistics and the timing.
  - Never committed: the fits' folders, weights, reads, logs and session files.

## 8. Self-tests, passed before this commit

`python outputs/mp_unified/zg6b2.py --selftest` runs on a toy carve and fake files in a temporary folder:
1. **The arm.** zg6b2 sits on P1c3's link carve, with zg6b's and zg6's classes below it. zg6b, zg6 and zspk are
   unchanged.
2. **Where.** train and read refuse anything but a Kaggle T4 on KG's image, and any name that is not G6b2's, before
   reading data.
3. **The start.** From seed 0, zg6b2 draws as zg6 and zg6b do. Its state is zg6b's plus c at 0 (121,809 own parameters
   at hidden 128), and its dropout generator starts where zg6b's does.
4. **At the start**, s_e, s_t and the shares equal zg6b's bit for bit: zg6's score and its twin's. g is 0.5 on every
   row, the score is zg6b's bit for bit, and base_z is never read.
5. **c at 0.** With a random a and b and c at 0, the gate on every row is zg6b's gate of its question, and the score is
   zg6b's, to rounding (the sigmoid runs on rows, not on questions).
6. **By hand.** With a random c, the in-degrees equal a count by python loops bit for bit. The score equals python
   loops over the questions and edges: zg6.by_hand for both passes, with the shares and in-degrees counted edge by
   edge.
7. **A row no edge reaches**, in a question with edges, has s_e = s_t bit for bit, a score equal to the twin's, and a
   gate equal to the question part alone. Such rows give a, b and c no gradient. A batch without the edges or the KT
   input stops.
8. **The dropout.** There is no draw at read. In training there is one draw from the module's generator:
   - the shares and the in-degrees count only the edges zg2.family_keep keeps;
   - the score equals the by-hand model over those edges;
   - the global generator moves as zg6b's does.
9. **Gradients** reach every parameter, including a, b and c.
10. **Toy fits** through lean_gpu's loop repeat bit for bit. The encoder, the stack, the readout and a, b and c all
    move.
11. **The gate beside.** From the toy's link arrays, each question's part, and each reached row's key, in-degrees and
    gate, equal the model's own and a brute count, row by row. g6b2_gate.json is gate_stats' output. An array with
    other bytes stops.
12. **Stamps and take.**
    - It refuses a fit that is not G6b2's, and a read on another device.
    - take copies records, reads, the gate file and the timing file, never models.pt, and leaves KG's and G6b's fits to
      them.
    - take refuses an unread fit, a name that is not G6b2's, and a file with other bytes.
13. **One device** across G6b2's fit, KG's two bases and G6b's zg6b.
14. **The verdict**, each outcome with its beside rows (the zg6 and zg6b re-calls, R@5, the MLP share, the gate and the
    timing):
    - ADOPT_FOR_FULL_RUN;
    - NOT_ADOPTED with G6r next: a GAIN and a LOSS; no GAIN; a GAIN that a raised floor turns WITHIN; a LOSS alone.

    INCOMPLETE in nine cases:
    - without KG's verdict file, or with KG's verdict not BASE_READY;
    - without G6b's verdict file, with G6b's verdict ADOPT_FOR_FULL_RUN, or NOT_ADOPTED with another next form;
    - without a compare;
    - with the fits on another device than KG's;
    - with a zg6 column that is not KG's zg6 read, or a zg6b column that is not G6b's zg6b read.
15. Nothing is written under G6b2's real folders.

`python outputs/mp_unified/zkg_session.py --selftest` also checks that both G6b2 names map to zg6b2.py, that the script
exists, and that a G6b2 fit queued twice stops. `python outputs/mp_unified/zkg.py --selftest` and
`python outputs/mp_unified/zg6b.py --selftest` pass unchanged.

The text encoder stays frozen. webqsp never trains. Test splits are never read. There is no LLM or transformer, and no
new graph.

## 9. Results

Not run yet.
