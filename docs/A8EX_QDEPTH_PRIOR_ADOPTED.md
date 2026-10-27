# A8ex: round 11's depth prior on the adopted arms, on both models

Declared 20 October 2026, before any number. The next form A8e's verdict named (zrckpqe and zspkqe, both NOT_ADOPTED
with no GAIN; docs/A8E_QDEPTH_PRIOR.md, sections 5 and 7, and `outputs/a8e/screen-verdict.json`, sha256 1150ffad…),
under docs/SCREENS.md section 3b and docs/VERDICT_AUDIT_2026_10_17.md (section 2, row 8). Code:
`outputs/mp_unified/za8ex.py`.

## 1. Why this, now

**A8e** learned round 11's depth prior inside the lean pair (zrckp and zspk) and added it to every row's score. Both
arms were NOT_ADOPTED with no GAIN and no LOSS. The adopted arms are R1's zrckpr (the MLP) and G5's zspx (the GNN).
Each was ADOPT_FOR_FULL_RUN at its screen (`outputs/r1/screen-verdict.json`, sha256 7561ebaa…;
`outputs/g5/screen-verdict.json`, af5bcf04…).

Below, each A8e read sits beside the adopted arm's own move on the same read. Both are R@5 minus the lean arm's card
fit, with the call after the seed-null re-call. Sources: A8e's re-calls
(`outputs/screen/scr-a8e-zrckpqe-pair-recall.json`, sha256 16c78935…; `scr-a8e-zspkqe-pair-recall.json`, e2f30362…)
and the adopted arms' (`scr-r1-zrckpr-pair-recall.json`, e3a90a3f…; `scr-g5-zspx-pair-recall.json`, 2342fa8d…).

**The MLP:**

| read (floor) | zrckpqe − zrckp (A8e) | zrckpr − zrckp (R1) |
| --- | --- | --- |
| L-musique, musique zero-shot (0.0720) | +0.0031 [−0.0020, +0.0083], WITHIN | +0.0076 [+0.0028, +0.0125]: GAIN filed, WITHIN under the null |
| L-musique, webqsp zero-shot (0.0102) | +0.0024 [−0.0087, +0.0135], WITHIN | +0.0135 [+0.0032, +0.0239], GAIN |
| L-musique, hotpotqa (0.0075) | −0.0068 [−0.0091, −0.0045], WITHIN | +0.0018 [−0.0003, +0.0038], WITHIN |
| L-musique, 2wiki (0.0075) | −0.0058 [−0.0075, −0.0041], WITHIN | +0.0014 [−0.0000, +0.0028], WITHIN |
| L-hotpotqa, musique in domain (0.0075) | +0.0002 [−0.0046, +0.0049], WITHIN | +0.0034 [−0.0015, +0.0086], WITHIN |

**The GNN:**

| read (floor) | zspkqe − zspk (A8e) | zspx − zspk (G5) |
| --- | --- | --- |
| L-musique, musique zero-shot (0.0720) | +0.0084 [+0.0023, +0.0147]: GAIN filed, WITHIN under the null | +0.0061 [+0.0006, +0.0119], WITHIN |
| L-hotpotqa, musique in domain (0.0075) | −0.0072 [−0.0127, −0.0014], WITHIN | +0.0147 [+0.0095, +0.0198], GAIN |
| L-hotpotqa, webqsp zero-shot (0.0233) | +0.0133 [+0.0030, +0.0232]: GAIN filed, WITHIN under the null | +0.0235 [+0.0133, +0.0327], GAIN |
| L-hotpotqa, hotpotqa zero-shot (0.0200) | +0.0053 [+0.0023, +0.0083], WITHIN | +0.0048 [+0.0020, +0.0076], WITHIN |

**A8e cannot say what the prior does on the adopted arms.**
- On the GNN, A8e's largest rise (webqsp zero-shot on L-hotpotqa, +0.0133) and its largest fall (musique in domain on
  L-hotpotqa, −0.0072) are zspx's two GAIN reads. From A8e alone, the prior could add to zspx's gains or cancel them.
- On the MLP, the prior's two largest falls, both on L-musique (hotpotqa −0.0068, 2wiki −0.0058), are reads where R1
  rose a little (+0.0018, +0.0014). R1's one GAIN under the null, webqsp zero-shot on L-musique (+0.0135), is a read
  where the prior moved +0.0024.

The prior has not yet met the adopted arms. A8e's section 5 fixed this file: the depth prior closes as a model term
only if A8ex finds no GAIN either.

**What changes against A8e.**
- **The bases.** zrckpr is zrckp plus R1's head over the row's own nine residual columns. zspx is zspk with its step's
  gradient kept, then G3's residual message and G4's second step.
- **What reads the prior in the GNN.** In zspkqe, zsp's one propagation read the score with the prior in it, under
  no_grad (zprop.prop_inputs). In zspxqe, zspx's step reads it with the gradient kept (zg4prop.prop_inputs_grad), so
  the prior also learns through the step. G3's residual message and G4's second step then read the score after the
  step.
- **Nothing else.** The prior's form, inputs, zero start and training are A8e's.

## 2. The form

A8e's, unchanged (round 11's form, `outputs/mp_unified/qdepth.py`):

    s(v) = s_base(v) + l_q[depth(v)],    l_q = q U + P_q W + b    (five values per question)

- **q** is the question's embedding: the P1c2 cache's `q_emb`, SEMB's input.
- **P_q** is its pool's profile (qdepth.profile): the share of its rows in each depth class, and log(1 + the pool's
  size). Six values.
- **depth(v)** is qdepth.depth_of's class of the row, from WALK's raw columns:
  - 0, a seed (column 15). A seed that is also a first hop is a seed.
  - 1 to 3, the first structural hop from a seed (columns 12 to 14). The nearest wins when a row carries two.
  - 4, unreached.
- **U** is 1,536 × 5, **W** 6 × 5, **b** 5: 7,715 parameters, all zero at the start. They draw nothing from any
  generator, so every other weight and the global random stream are the base's.
- **Training:** the model's listwise loss (lean_gpu's), with its own Adam, learning rate, weight decay and SWA (the card
  config 2e-3:1e-4:0.1:8:2). Only the differences between a question's five values move its ranking: the loss is a
  softmax over the question's rows.
- **A dropped block:** a question whose WALK or SEMB block is dropped takes no prior. Fits and reads keep every block,
  so this never acts here.
- **The code** is A8e's own (`za8e.qd_check`, `qd_init` and `qd_term`, through `qdepth.QDepth.depth_logits`). A8ex
  writes no prior of its own.

**No columns, no builder.** q and WALK are blocks of P1c2's cache that both adopted arms already read under variant p.
A8ex builds nothing and adds no rebuild.

**Where it enters.**
- **zrckprqe:** added to zrckpr's final score, after the KT head and R1's head. Nothing reads the score after it.
- **zspxqe:** added to the KT-mixed score by A8e's QDMix, before any of zspx's three increments reads the score. The
  step reads it first, with the gradient kept. G3's residual message reads the stepped score, and G4's second step reads
  the score after both.

**The MLP rule holds on zrckprqe.** The MLP never uses message passing, never reads another row's score, and never
reads anything normalised over the question's rows from a score.
- **What the prior reads:** the question's embedding and WALK's raw columns, both cached with the pool before any row
  is scored. It reads no label and no score, and passes nothing over graph neighbours.
- **What it normalises over the question's rows:** only P_q's shares, of a feature (the depth class), not of a score.
  It adds no z-score: every pool z-score zrckprqe takes is zrckpr's own, bit for bit.
- **Where it adds:** last. No later term reads the score, so no row's score reaches another row. A row's term depends
  only on its question's five values and its own class.
- **R1's columns do not change this.** R1's head reads the row's own nine columns (att, pmax and smax per link family).
  They are built from raw cosines before any row is scored (docs/R1_RESIDUAL_INPUT.md, section 2). The prior does not
  read them, and R1's head does not read the prior. The two terms add side by side, each from inputs alone.
- The self-test asserts each of these (section 6).

## 3. The arms

| arm | model | where the prior enters | carve |
| --- | --- | --- | --- |
| zrckprqe | zrckpr's model plus the prior (MRO: ZRCKPRQE, ZRCKPR, ZRCKP, KTMixP, ZRM, ChainMatch, ...) | added to the final score, after the KT head and R1's head | zrckpr's (`zr1res.R1KTChainCarveZRC`), unchanged |
| zspxqe | zspx's model with the prior in its score (MRO: ZSPXQE, ZSPX, ZSPR, ZSPK, ZProp, QDMix, KTMix, ZRM, ChainMatch, ...) | added to the KT-mixed score, before zspx's step reads it; G3's residual message and G4's second step read the score after the step | zspx's (`zg3res.ResKTLinkCarveBase`), unchanged |

zspxqe's class has no body: Python's method order puts A8e's QDMix between zspx's ZProp and the KT head (QDMix is not
in zspx's own). Each arm starts as its base, bit for bit: the base's parameters plus U, W and b at zero.

## 4. The fits and the call

A8e's, on the adopted arms (A2xmx's setup):
- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-a8ex-<arm>`, `-hp`), on P1c2's roots with
  P1c3's KT input and the base arm's own columns (R1's for zrckpr, G3's c and s for zspx), at the card fits' settings
  (variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split (zrckpr: scr-r1-zrckpr and -hp; zspx:
  scr-g5-zspx and -hp), step 1's fit beside. `compare` refuses any other base: the lean pair's fits, the other split's,
  the other arm's, or step 1's fit of another split.
- **Base screens for the re-call:** each base fit against step 1's. For zrckpr they are P1g-c's
  (`outputs/p1gc/base-pools-zrckpr-<split>.json`); for zspx, A9's (`outputs/a9/base-zspx-<split>.json`).
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), per arm. GAIN,
  LOSS and WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A8ex's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names each arm's next form (section 5).

**Reported beside, deciding nothing:**
- webqsp zero-shot, both splits and arms: where the hop-depth oracle doubled the MLPs' hit@1 (docs/SCREENS.md, 'Hop
  depth of the top-1 errors'), and where A8e's GNN rose most;
- musique in domain (L-hotpotqa), both arms side by side: Claim 1's read;
- metaqa, both splits and arms (the 3-hop questions the oracle was measured on);
- hit@1 and FC@5 per read, from the compare screens;
- each of the twelve reads beside A8e's on the lean pair, in the verdict (from A8e's two re-calls);
- per fit, the prior's norms |U|, |W| and |b| in each saved state (stamped by `train` as `prior_norms`).

**Serving cost** (reported later, not part of this call): as A8e's, per question one 1,536 × 5 product and the pool's
class counts, per row one add; the classes come from WALK, which both adopted arms already read.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools, then its full run, its own declared file |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A8exk, the back-off form on the adopted arms: the prior split by the graph's kind (one U, W and b for the KB graphs, metaqa and webqsp, and one for the passage graphs, each at zero), so one kind's depth weights cannot move the other kind's scores; on both models |
| NOT_ADOPTED with no GAIN, the other arm with a GAIN | A8exk on this model as well: the other arm's GAIN keeps the depth prior open on both models |
| NOT_ADOPTED with no GAIN, the other arm INCOMPLETE | decided with the other arm's re-call |
| NOT_ADOPTED with no GAIN on both arms | none for the depth prior as a model term, which closes (A8e's section 5) |
| INCOMPLETE | the missing re-call is run; nothing else changes |

**Why these.**
- **A GAIN with a LOSS** is a trade between reads. Round 11's pair lost webqsp zero-shot, a KB graph (−0.0204;
  docs/A8E_QDEPTH_PRIOR.md, section 1). A depth class means a hop along typed relations on a KB graph and a link
  between passages on a passage graph. Split by kind, each kind's weights learn where depth helps and move only that
  kind's scores. The kind is the graph's, known before any row is scored, so the MLP rule holds.
- **One arm's GAIN** keeps the idea open on both models, so the arm without a GAIN goes to the back-off as well
  (docs/SCREENS.md section 3b: an idea with signal reaches both models before it closes).
- **No GAIN on either arm:** the depth prior closes as a model term. It has then reached both models three ways
  without a GAIN: as a fixed input (A8), learned inside the lean pair (A8e) and learned inside the adopted arms (A8ex).
  **No non-model form remains.** The depth classes already reach both models through WALK, and the hop-depth
  diagnosis's pools already held a gold for 0.972 of metaqa's 3-hop questions and 0.921 of webqsp's, with only 0.001
  and 0.010 of their golds unreached (`outputs/diag/hopdiag-J5.json`, sha256 b816b316…); what is left is ranking by
  depth, which needs the question's hop count, and no rule without parameters can read that from a question on all
  six datasets. The hop-depth oracle's gap then goes into the gap ledger (docs/P3B_GAP_LEDGER.md) as an upper bound,
  not as a form.

No model change is adopted on today's pools (the user's rule of 13 October): an adopt is refitted on P1g-b's pools
before its full run.

## 6. How it runs

**Self-test.** `python outputs/mp_unified/za8ex.py --selftest` passes (5.0 s). It runs on A2xmx's toy carves (the KT
input, rmatch's chains, zlink's links, G3's c and s and R1's columns), in a batch of 16 pools of 3 to 9 rows, with
A8e's toy walks in WALK (all five classes, a seed that is also a first hop, rows with two first hops). It checks:
- **the arms:** the two MROs, with A8e's QDMix just above the KT head in zspxqe's and not in zspx's; zspxqe's class has
  no body; both arms use A8e's own prior function; the base arms and A8e's arms unchanged; qdepth's constants; 7,715
  added parameters; the bases are R1's and G5's card fits, and the base screens P1g-c's and A9's;
- **the start,** as built and again with every head's last layer set alike: from the same seed each arm is its base's
  state plus U, W and b at zero, the global random stream where the base leaves it. Each arm is its base bit for bit:
  - its scores, in training (the same dropout draws) and in eval;
  - every gradient of a first listwise step but the prior's. The prior's gradients reach U, W and b, finite and
    nonzero, and b's sum to 0 over the classes (to 1e-5);
- **the term,** with U, W and b set: it matches a float64 hand computation from the raw walks and embeddings (to
  1e-5). Then:
  - zrckprqe's score is zrckpr's plus the term, bit for bit. Every pool z-score it takes is the base's, bit for bit,
    and nothing propagates;
  - in zspxqe every z-score before the step is the base's. The step's first read of the score is the KT-mixed score
    plus the term. G3's residual reads the stepped score and G4's second step the score after it. The arm's score is
    rebuilt from them bit for bit, and it is not zspx's score plus the term: the increments read the prior;
- **never a row's score:** the term from WALK and the question's embedding alone is the same bit for bit. In
  zrckprqe it is the same without R1's columns, whose head moves the score. It stays the same when every other weight
  moves every score. A question whose WALK or SEMB is dropped takes none; the others keep theirs;
- **toy fits** through lean_gpu's loop repeat bit for bit and move U, W and b. With the prior frozen at zero, each
  arm's fit is its base's fit bit for bit on every shared key;
- **the refusals:** blocks without WALK or SEMB, or a WALK of another width; train refuses another arm (zrckpr,
  zrckpqe, zspx, zspkqe), another split, another arm's or split's fit name, and another cache root or device; read
  refuses a fit that is not A8ex's, not its arm's or not its split's; compare refuses a fit that is not A8ex's, the
  lean pair's base, another split's or arm's base, step 1's fit of another split or none, another screen name, a fit
  record of another split and a base record of another arm, and passes A8ex's own commands through unchanged; pair
  and re-call refuse another arm;
- relz's pair and re-call per arm against its base arm's card fits, stamped A8ex's, with relz restored after;
- the verdict's outcomes and their next forms: an adopt, a GAIN with a LOSS, no GAIN beside the other arm's GAIN, no
  GAIN on both arms (the text names A8, A8e and A8ex and the gap ledger), no GAIN beside an incomplete arm, and five
  kinds of INCOMPLETE (missing, another stage's, another arm's, another declared file's, eleven reads); A8e's re-calls
  beside each read, and "not found" when they are missing or another arm's.

**Commands:**

    python outputs/mp_unified/za8ex.py train --split L-musique --name scr-a8ex-zrckprqe --arm zrckprqe --device cuda \
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za8ex.py read --name scr-a8ex-zrckprqe --device cuda --host \
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za8ex.py compare --new outputs/screen/fits/scr-a8ex-zrckprqe \
        --base outputs/screen/fits/scr-r1-zrckpr,outputs/step1/fits/L-musique --out outputs/screen/scr-a8ex-zrckprqe
    python outputs/mp_unified/za8ex.py pair --arm zrckprqe \
        --screens outputs/screen/scr-a8ex-zrckprqe.json,outputs/screen/scr-a8ex-zrckprqe-hp.json \
        --out outputs/screen/scr-a8ex-zrckprqe-pair
    python outputs/mp_unified/za8ex.py recall --arm zrckprqe --null N1,N2,N3,N4 \
        --pair outputs/screen/scr-a8ex-zrckprqe-pair.json --out outputs/screen/scr-a8ex-zrckprqe-pair-recall
    python outputs/mp_unified/za8ex.py verdict

N1 to N4 are the four null files (`outputs/screen/scr-null-s1.json`, `-s2`, `-s1-hp`, `-s2-hp`). The same commands run
for L-hotpotqa (`--split L-hotpotqa`, names `-hp`, base scr-r1-zrckpr-hp and step 1's L-hotpotqa) and for zspxqe
(base scr-g5-zspx and -hp). The verdict writes `outputs/a8ex/screen-verdict.{json,md}`. On the host the trains and
reads run under `retry_cmd.py --cuda` and `cuda_alloc.py`, as A2xmx's.

**Host items** (17, `a8ex-*`), A2xmx's 17 renamed, at A2xmx's sizes:
- zrckprqe at R1's sizes: `--frac 0.30`, GPU share 0.32, 4.5 GB for the trains and 6 GB for the reads.
- zspxqe at A9's sizes for zspx: `--frac 0.40`, GPU share 0.42, 6 GB, and 15 GB for the L-hotpotqa train.
- The prior adds 7,715 parameters and one 1,536 × 5 product per batch, too little to move these sizes. They are not
  measured with the prior.
- zrckprqe's trains and reads wait on the n19-r1c-* rebuilds of R1's columns, zspxqe's on the n19-res-* rebuilds of
  G3's c and s, by A2xmx's lists. A8ex adds no rebuild item.
- zrckprqe's re-call waits on P1g-c's base screens (p1gc-base-pools-zrckpr-L-musique and -L-hotpotqa); zspxqe's waits
  on A9's (a9-base-zspx and -hp).
- Compares take 1 CPU and 4 GB; pairs, re-calls and the verdict take 1 CPU and 2 GB. The verdict also reads A8e's two
  re-calls.
- Inputs: A2xmx's, with za8ex.py, za8e.py and qdepth.py in place of za2xmx.py and za2xm.py, and
  configs/host_mirror_six.yaml, which `--host` reads (as A8e's items list it). The re-calls add the four null files.

**Disk:** about 0.3 GB of fits and reads, as A2xmx's.

The encoder stays frozen. webqsp never trains. Test splits are never read. No LLM, transformer or sentence-level
embedding, no new graph, and no GNN distillation.

## 7. Results

Filed 27 October 2026 from the host's records (`outputs/a8ex/screen-verdict.json`, written 2026-10-20T21:17:17Z; the
re-calls `outputs/screen/scr-a8ex-zrckprqe-pair-recall.json` and `scr-a8ex-zspxqe-pair-recall.json`, host paths made
relative). **zrckprqe ADOPT_FOR_FULL_RUN, zspxqe NOT_ADOPTED:** with round 11's depth prior added to zrckpr's final
score, the MLP gains webqsp zero-shot on L-hotpotqa and loses nothing; in zspx's score before its step reads it, the
prior moves no read past its floor.

- **zrckprqe: 1 GAIN, 0 LOSS.** The GAIN is webqsp zero-shot on L-hotpotqa: +0.0315 [+0.0211, +0.0430] against its
  0.0233 floor, with FC@5 +0.0273 and hit@1 +0.0160. Four more intervals exclude 0, all inside their floors: on
  L-musique, metaqa -0.0014 [-0.0022, -0.0005] and 2wiki -0.0029 [-0.0043, -0.0016]; on L-hotpotqa, musique in domain
  -0.0067 [-0.0117, -0.0010] and 2wiki -0.0026 [-0.0043, -0.0010]. No call changed under the re-call.
- **zspxqe: 0 GAIN, 0 LOSS.** Five intervals exclude 0, all inside their floors: on L-musique, 2wiki -0.0057 [-0.0072,
  -0.0042]; on L-hotpotqa, metaqa +0.0015 [+0.0006, +0.0024], musique in domain -0.0069 [-0.0120, -0.0016], hotpotqa
  zero-shot -0.0074 [-0.0103, -0.0043] and 2wiki +0.0029 [+0.0014, +0.0043]. webqsp zero-shot on L-hotpotqa moves
  -0.0089 [-0.0179, +0.0003]. No call changed under the re-call.
- **Beside: on the lean pair (A8e) every read of both arms was WITHIN.** webqsp zero-shot on L-hotpotqa moved +0.0030 on
  zrckpqe and +0.0133 on zspkqe; on the adopted arms it is the MLP's GAIN (+0.0315) and the GNN's -0.0089.

The verdict is per arm. **zrckprqe** is adopted for a full run; its next form, fixed in section 5 and recorded in the
verdict, is the arm fitted again on P1g-b's pools, then its full run, its own declared file. P1g-b's pools have since
been left out of the adopted models (P1g-c, docs/P1GC_ADOPTED_ON_P1GB.md section 8: the full run takes P1c2's pools), so
the pools of that refit are its own file's to fix. **zspxqe** is not adopted; this closes only its form, the prior in
zspx's score before its step (docs/SCREENS.md section 3b). Its next form, for no GAIN with the other arm's GAIN, is
A8exk on this model as well (section 5): the other arm's GAIN keeps the depth prior open on both models. A8exk is the
back-off form on the adopted arms: the prior split by the graph's kind (one U, W and b for the KB graphs, metaqa and
webqsp, and one for the passage graphs, each at zero), so one kind's depth weights cannot move the other kind's scores;
on both models.

The floors are the seed null of 7 October (docs/SCREENS.md section 2: `outputs/screen/scr-null-s1.json`,
`scr-null-s2.json` and their `-hp` files), not N2's fresh floors (docs/N2_FRESH_SEED_NULL.md).

Records, filed with this section: the verdict `outputs/a8ex/screen-verdict.json` and `.md`; for each arm (`zrckprqe`,
`zspxqe`), the compares `outputs/screen/scr-a8ex-<arm>.json` and `-hp.json`, the pair `-pair.json` and its re-call
`-pair-recall.json`, each with its `.md`. The host paths were made relative in each re-call's two `rel` entries and
nowhere else, so the `recall_sha256` each verdict holds is the host file's, from before that change. The re-calls' nulls
also read the base screens `outputs/a9/base-zspx-L-musique.json` and `outputs/a9/base-zspx-L-hotpotqa.json`, fetched
from the host with this filing (A9's own filing had left them there); and
`outputs/p1gc/base-pools-zrckpr-L-musique.json` and `outputs/p1gc/base-pools-zrckpr-L-hotpotqa.json`, filed with P1g-c's
results.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's, the base's card fit of the
split; a call is shown with its filed call where the re-call changed it):

| split | dataset | read | zrckpr R@5 | zrckprqe: delta [95% CI], call | zspx R@5 | zspxqe: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8012 | -0.0014 [-0.0022, -0.0005], WITHIN | 0.8030 | +0.0003 [-0.0006, +0.0012], WITHIN |
| L-musique | squad | in-domain | 0.9127 | -0.0017 [-0.0034, +0.0000], WITHIN | 0.9092 | +0.0017 [-0.0002, +0.0036], WITHIN |
| L-musique | musique | zero-shot | 0.5197 | -0.0002 [-0.0047, +0.0046], WITHIN | 0.5073 | +0.0027 [-0.0033, +0.0084], WITHIN |
| L-musique | hotpotqa | in-domain | 0.9083 | -0.0003 [-0.0022, +0.0016], WITHIN | 0.9074 | -0.0009 [-0.0032, +0.0011], WITHIN |
| L-musique | 2wiki | in-domain | 0.9361 | -0.0029 [-0.0043, -0.0016], WITHIN | 0.9351 | -0.0057 [-0.0072, -0.0042], WITHIN |
| L-musique | webqsp | zero-shot | 0.3762 | +0.0066 [-0.0030, +0.0160], WITHIN | 0.3204 | +0.0018 [-0.0049, +0.0084], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8040 | -0.0002 [-0.0010, +0.0006], WITHIN | 0.8039 | +0.0015 [+0.0006, +0.0024], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9114 | -0.0005 [-0.0025, +0.0015], WITHIN | 0.9129 | +0.0000 [-0.0021, +0.0022], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5779 | -0.0067 [-0.0117, -0.0010], WITHIN | 0.5901 | -0.0069 [-0.0120, -0.0016], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8558 | -0.0008 [-0.0039, +0.0020], WITHIN | 0.8547 | -0.0074 [-0.0103, -0.0043], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9331 | -0.0026 [-0.0043, -0.0010], WITHIN | 0.9312 | +0.0029 [+0.0014, +0.0043], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3411 | +0.0315 [+0.0211, +0.0430], **GAIN** | 0.2914 | -0.0089 [-0.0179, +0.0003], WITHIN |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a8ex-<arm>.json` and `-hp.json`); and A8e's R@5 deltas with the prior learned inside the lean pair
(zrckpqe against zrckp, zspkqe against zspk; their own re-calls, `outputs/screen/scr-a8e-<arm>-pair-recall.json`), on
the same read (a call is shown where it is not WITHIN):

| split | dataset | zrckprqe FC@5 | zrckprqe hit@1 | zspxqe FC@5 | zspxqe hit@1 | zrckpqe (A8e) R@5 | zspkqe (A8e) R@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L-musique | metaqa | -0.0012 | -0.0055 | +0.0014 | +0.0013 | -0.0000 | +0.0008 |
| L-musique | squad | -0.0017 | +0.0035 | +0.0017 | -0.0015 | -0.0008 | +0.0003 |
| L-musique | musique | +0.0070 | +0.0041 | +0.0000 | +0.0211 | +0.0031 | +0.0084 |
| L-musique | hotpotqa | -0.0004 | +0.0046 | -0.0003 | -0.0093 | -0.0068 | -0.0010 |
| L-musique | 2wiki | -0.0083 | -0.0044 | -0.0163 | +0.0037 | -0.0058 | -0.0042 |
| L-musique | webqsp | +0.0033 | -0.0033 | +0.0060 | -0.0080 | +0.0024 | -0.0007 |
| L-hotpotqa | metaqa | +0.0000 | +0.0007 | +0.0011 | +0.0030 | +0.0002 | -0.0008 |
| L-hotpotqa | squad | -0.0005 | -0.0019 | +0.0000 | +0.0000 | -0.0005 | -0.0024 |
| L-hotpotqa | musique | -0.0074 | -0.0029 | -0.0108 | -0.0095 | +0.0002 | -0.0072 |
| L-hotpotqa | hotpotqa | -0.0027 | -0.0063 | -0.0123 | -0.0143 | +0.0003 | +0.0053 |
| L-hotpotqa | 2wiki | -0.0050 | +0.0002 | +0.0077 | -0.0021 | -0.0040 | +0.0037 |
| L-hotpotqa | webqsp | +0.0273 | +0.0160 | -0.0047 | +0.0047 | +0.0030 | +0.0133 |
