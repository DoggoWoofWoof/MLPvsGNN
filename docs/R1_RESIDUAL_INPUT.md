# R1: the bridge residual as a fixed input to both models

Declared 17 October 2026, before any number. Code: `outputs/mp_unified/zr1res.py`. Two arms, one per model. Both read
the same nine columns, computed before either model runs.

Status: RUN 17 October 2026. zrckpr ADOPT_FOR_FULL_RUN, zspkr NOT_ADOPTED (section 6).

## 1. Why this, now

- **Claim 1 is not shown on musique.** The MLP keeps 0.933 of the GNN's R@5 gain there, with a 95% interval of
  [0.881, 0.983] (P3c, docs/WHERE_WE_STAND.md section 2). The bar is a lower end of at least 0.90.
- **The GNN's musique is moving up.** G3's residual message, G4's second step and G4's end-to-end step each lift
  zspk's musique in-domain read by about 0.01 (L-hotpotqa), and G5 combines them. Each of those is a message, and the
  MLP takes no message (the user's rule of 8 October). If the GNN gains and the MLP does not, musique's share falls
  further. WHERE_WE_STAND section 7 already says so: "the MLP's inputs must catch up".
- **What musique misses.** M1 and D1 found the missed golds one or two links from found ones (REACHABLE_NOT_RANKED).
  D5b found that the residual cos(q − (q·p̂)p̂, v) orders a found node's neighbours about as well as the cosine does.
  So the residual carries information, but no model reads it before scoring.
- **What the MLP may read.** The MLP's scoring has no message passing: no row's score reaches another row, and no
  input is a score normalised over the question's rows (8 and 14 October). It does read fixed inputs computed from the
  graph before any row is scored. KIND_T's walk positions, P1c3's KT input, are one such input. R1 adds another: the
  residual, aggregated over each row's pool neighbours, with no score and no learned weight.
- **Both models get it** (F1, docs/F1_ONE_INPUT_SET.md: one input set, read by both). Each arm is decided against its
  own model's card fit, never against the gap between the two. G3's result is why the residual is tested now. No GNN
  number chooses whether the MLP keeps it.

## 2. The input

ê is the frozen encoder's dense embeddings made unit length: the question's from the look, the pool rows' from the
package. The pieces are G3's builds, unchanged (`outputs/g3/res`, docs/G3_BRIDGE_RESIDUAL.md section 2):

    c_v  = ê_q · ê_v            each row's cosine to the question
    s_uv = ê_u · ê_v            each pool edge's cosine (zlink's edges, in its order)
    r_uv = (c_v − c_u s_uv) / sqrt(max(1 − c_u², 1e-4)), clamped to [−1, 1]

**At each row v, per link family f** (the three of zlink), over v's pool neighbours u in f, with each parent's weight
w_u = exp(τ c_u), τ = 10:

    att  = Σ w r / Σ w                          the cosine-weighted mean residual
    pmax = (log Σ w exp(τ r) − log Σ w) / τ     the cosine-weighted soft maximum
    smax = log Σ exp(τ r) / τ                   the unweighted soft maximum

- Each is 0 where v has no parent in f. That makes nine columns: att, then pmax, then smax, each over the three
  families.
- These are G3's att, pmax and smax with one change. G3 weighs a parent by exp(z_u), z_u being the GNN's z-scored
  score. R1 weighs it by exp(τ c_u), the parent's raw cosine to the question. It is not normalised over the question
  and is known before any row is scored.
- The columns are computed once per cache part on the host (`build`), from G3's c and s and zlink's edges, in float64.
  They are stored as float16. There is no parameter, no label and no score.

**A worked example** (G3's). "Who is the spouse of the director of *Film X*?"
- The page of *Film X* (u) is in the pool: c_u = 0.60.
- v₁ is a page about the film's music: c_v = 0.35, s_uv = 0.50, so r = 0.06.
- v₂ is the director's page: c_v = 0.35, s_uv = 0.10, so r = 0.36.
- With u as the only parent in that family, all three columns are r: v₁ reads 0.06 and v₂ reads 0.36. A model that
  reads only c_v sees two equal rows. With R1's columns it can tell them apart, before any score exists.

## 3. The arms

| arm | model | carve | its base |
| --- | --- | --- | --- |
| zrckpr | zrckp plus the R1 head (`ZRCKPR`) | zrckp's (`KTChainCarveZRC`) with R1's columns beside | `scr-p1c3p-zrckp`, `-hp` |
| zspkr | zspk plus the R1 head, added with the KT head before zsp's propagation reads the score (`ZSPKR`) | zspk's (`KTLinkCarveBase`) with R1's columns beside | `scr-p1c3-zspk`, `-hp` |

- **The head** is the same in both: Linear(9, 16), GELU, Linear(16, 1), over the row's own nine columns, added to the
  score. That is 177 parameters, the shape of zrckp's pointwise KT head. It has no score input, so it is pointwise in
  both models.
- Its last layer starts at zero, so each arm starts as its base bit for bit. Its first layer is drawn from its own
  generator (seed + 5501).
- Everything else is the base's, unchanged: the carve, KT input, settings, seeds, batches, loss and loop. Neither arm
  adds a graph, an encoder, text or a message. The MLP's carve gets the nine columns and no edge.

## 4. The fits and the call

- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-r1-<arm>`, `-hp`), on P1c2's roots,
  P1c3's KT input and R1's columns.
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split, which decides. Step 1's fit is reported
  beside. The re-call's base screens are L1's (`outputs/l1/base-<base arm>-<split>.json`).
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), per arm. GAIN,
  LOSS and WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not R1's re-call of that arm, or does not hold the twelve reads.

**What follows.** No model change is adopted on today's pools (the user's rule of 13 October). An arm that passes is
fitted again on P1g-b's pools, which needs G3's and R1's builds on P1g-b's parts. That refit and its full run are
their own file, declared before their numbers. If both arms pass, the input is the lean pair's, read by both. If only
one passes, the record says which, and the next file decides whether F1's one-input-set rule keeps it.

**Reported beside, deciding nothing:**
- webqsp (zero-shot in both splits) and metaqa (in-domain), per split and arm, with their calls and differences;
- musique's in-domain read (L-hotpotqa) for both arms side by side: the MLP's change and the GNN's. This is what
  Claim 1's musique share depends on;
- hit@1 and FC@5 per read, as the compares give them.

**Flagged.** Like KIND_T's walks, the columns read the pool's edges. They are computed before either model runs and
are the same for both, so the MLP's scoring stays free of message passing: no row's score, and no value normalised
over the question, reaches another row. Their serving cost is the per-edge cosines s_uv (one dot product per pool
edge) and three sums per row. Both models share it, and it is timed cold beside P4's timings before any latency claim.

## 5. How it runs

- **Self-test** (`python outputs/mp_unified/zr1res.py --selftest`, passes before this commit):
  - the arms are zrckp's and zspk's models plus the head, on their carves with R1's columns beside. The base arms are
    unchanged, and zspkr's MRO puts the head before zsp's propagation;
  - the columns equal a float64 reference built row by row and edge by edge. They are 0 where a row has no parent.
    With one parent, all three equal r, and r is the geometric cos(q − (q·û)û, v);
  - on a toy built through G3's own build: R1's build refuses a G3 record of another part and zlink edges that are not
    G3's. A batch's columns are the rows' own (float16). The MLP's carve holds no edge;
  - each arm starts as its base bit for bit. With the head moved, the difference is the head over the batch's
    columns. Without the columns it is its base. In zspkr the propagation reads the score with the head added;
  - toy fits through lean_gpu's loop: a repeat fit is bit-identical and the head learns. Without the columns, each
    arm's fit equals its base's bit for bit on every shared key, with the head at its start;
  - train refuses another arm, split, fit name, cache root or device. read refuses a fit that is not R1's;
  - relz's pair and re-call per arm against its base arm's card fits, stamped R1's; the verdict's outcomes per arm.
- **Host items** (`r1-*`, placed after G5's verdict and ahead of P1g-b's remaining items):
  - on the CPU: 39 builds, one per P1c2 cache part (G3's 39, which are on the host), 1 CPU and about 3 GB each;
  - on the card: four trains, then four reads, at `--frac 0.30` and share 0.32 (L1's), at most three at once;
  - on the CPU: four compares, two pairs, two re-calls and the verdict.
- **Disk:** 53.7 million rows across the 39 parts, nine float16 columns each: about 1.0 GB.
- **ETA:** the 39 parts hold 751 million edges. The builds read about 1.5 GB of G3's arrays and 3.8 GB of zlink's,
  about 15 to 30 minutes in all. E1's fit of zspk's model took 47 minutes alone on the card. Three fits at once, then
  the fourth, then the reads: about 2.5 to 3 hours, so the verdict is due about 14:00 on 17 October. P1g-b's fits are
  not ready before then.
- The encoder stays frozen. webqsp never trains. Test splits are never read. No LLM or transformer.

## 6. Results

Run 17 October 2026 on the host: the 39 builds, four fits, four reads, compares, pairs, re-calls and the verdict (`outputs/r1/screen-verdict.json` and `.md`, `outputs/screen/scr-r1-*`).

**Verdict: zrckpr ADOPT_FOR_FULL_RUN, zspkr NOT_ADOPTED.**

R@5 of each arm minus its base's card fit of the same split (95% question bootstrap). The call is the re-call under the seed null; a call the null changed is marked with its filed call.

| split | dataset | read | zrckpr − zrckp | call | zspkr − zspk | call |
| --- | --- | --- | ---: | --- | ---: | --- |
| L-musique | metaqa | in-domain | -0.0007 [-0.0016, +0.0002] | WITHIN | +0.0017 [+0.0008, +0.0026] | WITHIN |
| L-musique | squad | in-domain | +0.0011 [-0.0006, +0.0028] | WITHIN | +0.0021 [+0.0000, +0.0041] | WITHIN |
| L-musique | musique | zero-shot | +0.0076 [+0.0028, +0.0125] | WITHIN (filed GAIN) | -0.0079 [-0.0141, -0.0017] | WITHIN (filed LOSS) |
| L-musique | hotpotqa | in-domain | +0.0018 [-0.0003, +0.0038] | WITHIN | -0.0005 [-0.0026, +0.0017] | WITHIN |
| L-musique | 2wiki | in-domain | +0.0014 [-0.0000, +0.0028] | WITHIN | -0.0041 [-0.0055, -0.0027] | WITHIN |
| L-musique | webqsp | zero-shot | +0.0135 [+0.0032, +0.0239] | GAIN | -0.0005 [-0.0089, +0.0077] | WITHIN |
| L-hotpotqa | metaqa | in-domain | -0.0004 [-0.0012, +0.0005] | WITHIN | -0.0016 [-0.0024, -0.0006] | WITHIN |
| L-hotpotqa | squad | in-domain | -0.0013 [-0.0033, +0.0006] | WITHIN | -0.0003 [-0.0021, +0.0017] | WITHIN |
| L-hotpotqa | musique | in-domain | +0.0034 [-0.0015, +0.0086] | WITHIN | +0.0011 [-0.0042, +0.0064] | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | +0.0045 [+0.0016, +0.0074] | WITHIN | -0.0032 [-0.0061, -0.0003] | WITHIN |
| L-hotpotqa | 2wiki | in-domain | -0.0019 [-0.0034, -0.0003] | WITHIN | -0.0009 [-0.0025, +0.0007] | WITHIN |
| L-hotpotqa | webqsp | zero-shot | -0.0064 [-0.0156, +0.0028] | WITHIN | +0.0078 [-0.0012, +0.0169] | WITHIN |

- **zrckpr:** 1 GAIN (L-musique webqsp), 0 LOSS (none) of twelve after the re-call.
- **zspkr:** 0 GAIN (none), 0 LOSS (none) of twelve after the re-call.

**Reported beside, deciding nothing.**
- musique in domain (L-hotpotqa), Claim 1's read: the MLP +0.0034, the GNN +0.0011, both WITHIN. R1 moves the MLP's musique at least as much as the GNN's, but neither change is above the floor, so Claim 1's musique share is not recomputed from this screen.
- webqsp zero-shot (L-musique), the MLP's GAIN: R@5 +0.0135, FC@5 +0.0073, hit@1 +0.0106. The GNN's read there: -0.0005.
- metaqa in domain: zrckpr −0.0007 and −0.0004, zspkr +0.0017 and −0.0016, all WITHIN.
- The null turned one filed call each: zrckpr's L-musique musique GAIN (+0.0076) and zspkr's L-musique musique LOSS (−0.0079) are both WITHIN its floor of 0.0720.

**This verdict closes only this form** (docs/SCREENS.md section 3b): R1's nine columns through a pointwise head, on P1c2's pools, for zrckp and zspk.

**What follows**, each in its own file, declared before its numbers:
- **zrckpr (ADOPT_FOR_FULL_RUN):** refit on P1g-b's pools, which needs G3's and R1's builds on P1g-b's parts, then its full run (section 4's "What follows"). Only the MLP passed, so that file also decides whether F1's one-input-set rule keeps the input: R1's columns go beside zspx's inputs in the same full run, so both models read them and the GNN can learn to leave them at zero.
- **zspkr (NOT_ADOPTED, no GAIN and no LOSS):** on zspk the pointwise copy moved nothing past the floor. zspx, the
  GNN going on to P1g-b's pools, already reads the residual as G3's message, and R1's columns reach it in zrckpr's
  full run (above). The audit's conditional item now runs: G1b's bridge columns (zbr's) on zspk,
  docs/A13_BRIDGE_COLUMNS_GNN.md (docs/VERDICT_AUDIT_2026_10_17.md, section 2).
