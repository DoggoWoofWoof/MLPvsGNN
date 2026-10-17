# G3: bridge-residual messages for the GNN

Declared 17 October 2026, before any number. Code: `outputs/mp_unified/zg3res.py`. One arm, the GNN's only. The MLP
takes no message (the user's rule of 8 October).

## 1. Why this, now

The user (15 October): the GNN "should exploit message passing and other stuff as much as possible", and musique needs
to be better. What we know:
- **musique's chain breaks at the bridge** (M1, docs/M1_MUSIQUE_HOPS.md section 6). The first paragraph is found (0.70
  to 0.84 in the top five), and the next one is usually one or two links from it (0.56 of musique's missed golds; 0.50
  on hotpotqa, 0.69 on 2wiki).
- **zspk lifts those golds no more often than the MLP** (a net +46 of musique's golds, −31 on hotpotqa, +111 of 30,654
  on 2wiki). Its one propagation step reads only its neighbours' scores. A row learns "a neighbour scored high". It
  does not learn "this neighbour leads here for the part of the question it does not already cover".
- **G2** gave the GNN hidden-state messages (three layers with attention and family dropout). It was not adopted
  (0 GAIN, 0 LOSS against zspk). More of the same message does not help, so G3 changes what the message says.
- **D5b and P1g** used the residual cos(q − (q·p̂)p̂, v) to order a found node's neighbours. As a pool order it reaches
  about as many golds as the cosine (musique 0.913 for both at k = 1,000, D5b), and P1g's own pick used it. No model
  has read it yet.

G3 gives the GNN that residual as a message, weighted by how well each parent scores. It needs no parameter and no new
graph, and it uses only the frozen encoder's embeddings, which the first stage already uses.

## 2. The message

With ê the frozen encoder's dense embeddings made unit length (the question's from the look, the pool rows' from the
package):

    c_v  = ê_q · ê_v            each row's cosine to the question
    s_uv = ê_u · ê_v            each pool edge's cosine (zlink's edges, in its order)
    r_uv = (c_v − c_u s_uv) / sqrt(max(1 − c_u², 1e-4)), clamped to [−1, 1]

r_uv is the cosine of v with the question once u's direction is taken out of it: cos(q − (q·û)û, v). It is high when v
covers the part of the question that u does not.

**A worked example.** "Who is the spouse of the director of *Film X*?"
- The page of *Film X* (u) is found: c_u = 0.60.
- Candidate v₁ is a page about the same film's music: c_v = 0.35, s_uv = 0.50. Then
  r = (0.35 − 0.60·0.50)/0.80 = 0.06. It repeats u.
- Candidate v₂ is the director's page, which names a spouse: c_v = 0.35, s_uv = 0.10. Then
  r = (0.35 − 0.06)/0.80 = 0.36. It covers the rest of the question.
- Both have the same cosine to the question, so a score that reads c_v alone cannot tell them apart. The residual can.

**At each row v, per link family f** (the three of zlink), with each parent u's weight w_u = exp(z_u):

    att  = Σ w r / Σ w                          the score-weighted mean residual
    pmax = (log Σ w exp(τ r) − log Σ w) / τ     the score-weighted soft maximum, τ = 10
    smax = log Σ exp(τ r) / τ                   the unweighted soft maximum

- z_u is the parent's z-scored score in its question (zspk's full score), clipped to ±8, with no gradient.
- Each of the three is 0 where v has no parent in f.
- With v's own z, that makes ten inputs (three per family plus z).

**The head:** Linear(10, 32), GELU, Linear(32, 1).
- Its last layer starts at zero, so zspr starts as zspk bit for bit. Its first layer is drawn from its own generator
  (seed + 3401).
- That is 385 parameters, added to zspk's score after zspk's score is complete.
- The inputs carry no gradient, so the head learns only how to weigh them.

**The arm:**

| arm | model | carve | its base |
| --- | --- | --- | --- |
| zspr | zspk + the residual head (`ZSPR`) | zspk's carve with c and s beside (`ResKTLinkCarveBase`) | `scr-p1c3-zspk`, `-hp` |

Everything else is zspk's, unchanged: the settings, seeds, batches, KT input, loss and loop.

## 3. The fits and the call

- **Build** (CPU, on the host, one item per step-1 cache part of P1c2's roots, 39 in all):
  - It reads each look chunk's pool node ids and question embedding, and the package's dense docs embeddings. The
    canonical freeze is pinned (RECORD_SHA256 58958f33…), and the build refuses on a mismatch.
  - It writes e_s per edge and r_c per row, as float16.
  - Each part's edges are checked equal to zlink's arrays. Each chunk's first question checks the embeddings read
    against the look's own projection of them (its seed and width; off by at most 2e-3).
  - Beside, deciding nothing, each part's record carries the mean best-parent residual of gold rows and of the others.
- **Fits:** one card fit on L-musique and one on L-hotpotqa (`scr-g3-zspr`, `-hp`), on P1c2's roots and P1c3's KT
  input, as zspk's were.
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against zspk's card fit of the same split, which decides. Step 1's fit is reported beside.
  The re-call's base screens are L1's (`outputs/l1/base-zspk-<split>.json`).
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md). GAIN, LOSS and
  WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

- **ADOPT_FOR_FULL_RUN:** zspr's re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not G3's re-call of zspr, or does not hold the twelve reads.

Under the user's rule of 13 October, no model change is adopted on today's pools. If zspr passes, it is fitted again
on P1g-b's pools before its full run (six splits under grade-nullx, its own file).

**Reported beside, deciding nothing:**
- webqsp (zero-shot in both splits) and metaqa (in-domain), per split, with their calls and differences;
- hit@1 and FC@5 per read, as the compares give them;
- the builds' best-parent residual of gold rows against the others, per part.

## 4. How it runs

- **Self-test** (`python outputs/mp_unified/zg3res.py --selftest`, passes before this commit):
  - the arm is zspk's model plus the head, on zspk's carve with c and s beside. zspk's arm is unchanged;
  - r equals the geometric cos(q − (q·û)û, v) on 50 random triples (to 1e-5);
  - `res_inputs` matches a float64 reference built edge by edge (to 1e-4). Rows without parents read 0;
  - on rmatch's toy look, given pool node ids, question embeddings and the look's projection, the build:
    - refuses a node table that is not the look's (the projection check);
    - refuses zlink edges of another step-1 part, and a part that does not exist;
  - a batch's c and s equal each row's and edge's own, from the node table by hand (float16);
  - at the start, zspr's forward equals zspk's bit for bit. With the head moved, the difference is the head over
    `res_inputs` of zspk's score. Without c and s, it is zspk's forward;
  - toy fits through lean_gpu's loop:
    - a repeat fit is bit-identical, and the head learns;
    - on the carve without c and s, zspr's fit equals zspk's fit bit for bit on every shared key, in every epoch and
      in SWA, and its head stays at its start;
  - train refuses another arm, split, fit name, cache root or device. read refuses a fit that is not G3's;
  - relz's pair and re-call against zspk's card fits, stamped G3's; the verdict's three outcomes.
- **Host items** (`g3-*`, 48 in all):
  - 39 builds on the CPU (1 CPU, 4 GB each), through `pylib_run.py`, placed ahead of P1g-b's look shards. P1c2's
    roots are on the host, so they need nothing else;
  - on the card: two trains, then two reads (`--frac 0.40`, gpu share 0.42). Each train waits for its split's
    builds, and each read for its train and the six s1eval carves' builds. They run behind E1's and T1's items;
  - on the CPU: two compares, the pair, the re-call and the verdict.
- **Disk:** about 1.5 GB for e_s and r_c (two bytes per edge and per row; P1c2's links hold about 700 million edges).
  P1c2's roots stay until G3's reads are done, as for E1 and T1.
- **ETA:** the builds take about 6 CPU-hours (a quarter-second per musique question), about an hour of wall time beside
  P1g-b's looks. The fits and reads take about 3 hours on the card, as zspk's did. About 4 to 5 hours in all, in the
  card's gap between T1's reads and P1g-b's fits.
- The encoder stays frozen. webqsp never trains. Test splits are never read. No LLM or transformer.

## 5. Results

Filed 17 October 2026 from the host's records (`outputs/g3/screen-verdict.json`, written 2026-10-17T02:33:59Z; the re-calls `outputs/screen/scr-g3-zspr-pair-recall.json`). **zspr ADOPT_FOR_FULL_RUN.**

- **zspr: 1 GAIN, 0 LOSS.** musique in-domain rises 0.0096 on L-hotpotqa ([+0.0043, +0.0148]).
- On L-musique, zero-shot musique rises 0.0114 ([+0.0056, +0.0174]). That is filed a GAIN but re-called WITHIN under
  that read's floor (0.072). Both musique reads move up, which is the dataset the residual was built for (section 1).
- **Beside:**
  - webqsp zero-shot is −0.004 on L-musique and +0.013 on L-hotpotqa (both WITHIN);
  - metaqa in-domain is +0.001 and +0.000;
  - hotpotqa zero-shot is −0.003 (WITHIN).

The residual message lifts musique's ranking where zspk's score message did not (M1). zspr goes on, per section 3:
with G4's two arms, which also passed, it is combined in G5 (docs/G5_COMBINED_GNN.md, declared after these verdicts).
The GNN that goes to P1g-b's pools is G5's if G5 passes, otherwise each passing change on its own.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's; a call is shown with its filed call where the re-call changed it):

| split | dataset | read | base R@5 | zspr: delta [95% CI], call |
| --- | --- | --- | ---: | --- |
| L-musique | metaqa | in-domain | 0.8009 | +0.0012 [+0.0003, +0.0020], WITHIN |
| L-musique | squad | in-domain | 0.9114 | -0.0004 [-0.0024, +0.0015], WITHIN |
| L-musique | musique | zero-shot | 0.5012 | +0.0114 [+0.0056, +0.0174], WITHIN (filed GAIN) |
| L-musique | hotpotqa | in-domain | 0.9080 | +0.0003 [-0.0017, +0.0024], WITHIN |
| L-musique | 2wiki | in-domain | 0.9383 | -0.0018 [-0.0031, -0.0005], WITHIN |
| L-musique | webqsp | zero-shot | 0.3229 | -0.0038 [-0.0128, +0.0046], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8040 | +0.0003 [-0.0006, +0.0012], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9135 | -0.0003 [-0.0023, +0.0015], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5754 | +0.0096 [+0.0043, +0.0148], **GAIN** |
| L-hotpotqa | hotpotqa | zero-shot | 0.8499 | -0.0029 [-0.0057, +0.0000], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9333 | +0.0025 [+0.0010, +0.0039], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2680 | +0.0132 [+0.0025, +0.0235], WITHIN (filed GAIN) |
