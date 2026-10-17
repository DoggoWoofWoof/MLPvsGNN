# A13: the first stage's bridge columns on the GNN

Declared 17 October 2026, before any number. The thirteenth reopening of docs/VERDICT_AUDIT_2026_10_17.md: its
conditional item (section 2), run because R1's GNN arm (zspkr) was NOT_ADOPTED. Under docs/SCREENS.md section 3b.
Code: `outputs/mp_unified/za13br.py` (the arm, the screen and the call). Not run yet.

## 1. Why this, now

**Where the idea came from.** G1a (9 October, docs/G1A_UNIVERSAL_BRIDGE_SIX.md) tested a training-free bridge rule: keep
RRF's top 3, and fill slots 4 and 5 with the structural neighbours of RRF's top 2 that have the highest cos(q). Against
RRF's top 5 (R@5, x100):

| dataset | RRF | the rule | gain [95% CI] | call |
| --- | ---: | ---: | --- | --- |
| metaqa | 0.5 | 11.8 | +11.3 [+10.8, +11.9] | ABOVE |
| squad | 90.5 | 88.3 | -2.2 [-2.5, -1.9] | BELOW |
| musique | 47.3 | 47.6 | +0.2 [-0.6, +1.1] | SAME |
| hotpotqa | 68.5 | 80.1 | +11.6 [+11.1, +12.2] | ABOVE |
| 2wiki | 60.8 | 68.4 | +7.6 [+7.1, +8.0] | ABOVE |
| webqsp | 5.3 | 12.2 | +6.9 [+5.7, +8.1] | ABOVE |

A fixed rule gives up slots 4 and 5 on every question, single-hop ones too, so it loses on squad. A model that reads the
rule's quantities can decide per question.

**What it did so far.**
- **The MLP (round 29, zbr on zrc):** NO_GAIN. All twelve reads WITHIN; musique zero-shot +0.0053, musique in domain
  +0.0015. The MLP's form is closed (the audit, section 4).
- **Both models through R1 (17 October):** R1 carried the bridge signal as a residual input, through a pointwise head.
  The MLP passed (zrckpr ADOPT_FOR_FULL_RUN). The GNN did not (zspkr NOT_ADOPTED, 0 GAIN, 0 LOSS).

**Why that does not close the idea.** zbr's 13 columns never reached the GNN. R1's residual is a different quantity, a
score difference, not the rule's links. The GNN's step reads its neighbours' scores, so it can use what the MLP could
not: once the bridge head shifts a row's score, the step carries that shift to the row's neighbours. Under section 3b,
an idea stays open until it reaches both models. This is the GNN's form.

**The base is zspk,** as the audit names, decided against zspk's card fits.

## 2. The arm

**zspkb** is zspk (zp1c3.ZSPK on P1c3's KT link carve) whose score adds zbr's head over zbr's 13 inputs, before zsp's
propagation step reads the score (R1's placement):

    s  = zrm's score + the KT head          (zspk's, unchanged)
    s' = s + head(bridge_inputs(rrf's order, the pool's links, dense_cos))
    out = s' + zsp's step over s'           (zspk's step, unchanged)

**The inputs** are `zbr.bridge_inputs`, unchanged. Per row:
- log1p of its links from RRF's top 1, top 2 and top 5, per link family (9);
- log1p of its two-step paths from RRF's top 5, over any family (1);
- the bridge flag: linked to RRF's top 2 and not one of them (1);
- the bridge rank: 1 / (1 + the row's rank by dense_cos among its question's flagged rows), 0 when not flagged (1);
- the source rank: 1 / (1 + the best RRF rank among its neighbours), 0 when it has none (1).

They are fixed before any row is scored: RRF's order (the batch's base_z), zlink's pool edges, and the row's cached
dense_cos column. The head reads no score.

**The head** is zbr's: Linear(13, 32), GELU, Linear(32, 1), 481 parameters. The first layer is drawn from its own
generator (seed + 2901, zbr's draw); the last starts at zero. So at the start zspkb is zspk bit for bit, and no draw of
zspk's moves.

**The carve** is zspk's (P1c3's KT link carve) with each row's dense_cos beside its links, as zbr's carve adds it. Every
block, row, edge, KT column and gold is unchanged. It subclasses zspk's carve, so P1c2's roots (patched on zlink's and
rmatch's carves) reach it. zbr's own carve function would build a fresh link class and miss them, so it is not used.
No build.

**GNN only.** The MLP's form closed on zrc. The MLP's lean model has changed since (zrckp), so a gain here brings the
columns to zrckp next (section 5). zbr's inputs are legal there: none reads a score.

| arm | the model | the carve | added |
| --- | --- | --- | --- |
| zspkb | zspk + zbr's head before the step | zspk's + dense_cos beside the links | 481 parameters |

## 3. Cost

zbr's card fits took 2,777 s (L-musique) and 2,984 s (L-hotpotqa), against zrc's 2,166 s and 2,114 s: 1.28 and 1.41
times, at the same peak GPU memory (3.5 and 5.2 GB). Part of that is loading zlink's links, which zspk already pays.

zspk's card fits took 2,476 s and 2,559 s, at peaks of 3.7 and 5.6 GB. So each zspkb fit should take at most about
3,200 to 3,600 s, at zspk's memory.

Serving: per question, two sorts of its pool and three passes over its edges, all from fixed inputs. Any latency figure
for zspkb is cold, timed in its own declared stage.

## 4. The fits and the call

- **Fits:** one card fit on L-musique and one on L-hotpotqa (`scr-a13-zspkb`, `-hp`), on P1c2's roots with P1c3's KT
  input. Card fit settings: variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128 (zspk's).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against zspk's card fit of the same split (scr-p1c3-zspk, scr-p1c3-zspk-hp), which decides.
  Step 1's fit is reported beside.
- **The re-call's base screens:** L1's, zspk's card fits compared with step 1's of the same split
  (outputs/l1/base-zspk-<split>).
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md). GAIN, LOSS and
  WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

- **ADOPT_FOR_FULL_RUN:** zspkb's re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A13's re-call of zspkb, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- webqsp and metaqa, both splits (webqsp never trains; metaqa trains on both splits);
- musique in domain (L-hotpotqa): Claim 1's read;
- squad in domain, the one dataset where G1a's rule was BELOW;
- hit@1 and FC@5 per read.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | zspkb fitted again on P1g-b's pools, then its full run beside zspk's; and A13m, zbr's head on zrckp (the MLP's lean model), since zbr's MLP form closed on zrc before the MLP's model changed (each its own declared file) |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A13b, the back-off form: the bridge head bounded (0.2 tanh(h / 0.2), the MP-unified track's bound), so the step cannot lean on it where the rule fails (squad, comparison questions) (one declared file) |
| NOT_ADOPTED, with no GAIN | A13x, zbr's bridge head on zspx (the adopted GNN, three steps, where depth paid), before its first step (one declared file) |
| INCOMPLETE | the missing re-call is run; nothing else changes |

No model change is adopted on today's pools (the user's rule of 13 October).

## 6. How it runs

**Self-test, passed before this commit.** `python outputs/mp_unified/za13br.py --selftest` (7 s) checks:
- **The arm.** zspkb is zspk plus zbr's head (zbr's shape, sizes and seed offset, distinct from zspk's heads') on
  zspk's carve with dense_cos beside the links; MRO ZProp, BrMix, KTMix, ZRM. zspk's, zsp's and zbr's own arms are
  unchanged. P1c2's roots reach A13's carve.
- **The carve,** on P1c3's toy KT link carve: its batch is zspk's (rows, links, KT columns, base_z and golds), plus each
  row's cached dense_cos column.
- **The inputs** are zbr's by hand (Python sorts and loops over each question's edge list).
- **The start.** From one seed, zspkb is zspk's state plus zbr's head as zbr draws it, the last layer zero; the global
  generator ends where zspk leaves it. Its forward is zspk's bit for bit, in eval and in training with the same dropout
  draws, also with zspk's step and KT heads moved.
- **The placement.** With the head moved, zspkb is zspk's step over (zspk's pre-step score + zbr's head): the step reads
  the bridge-shifted score. Without links it is zspk's pre-step score; links without dense_cos are refused.
- **Toy fits** through lean_gpu's loop: a repeat is bit for bit; the head trains; through the first step the shared
  weights match zspk's own fit (the head adds 0 there), and every state after it differs.
- **The refusals.** train refuses zspk's arm, zbr's arm, another split, the wrong fit name, another cache root and the
  CPU. read refuses zspk's fit, zbr's fit and an unstamped fit.
- **The rest.** relz's pair and re-call stamped A13's, with relz restored after; the verdict's outcomes and next forms.

**Host items** (`a13-*`), after A12's verdict in the file, each starting when the GPU frees:
- 2 trains and 2 reads, sized from zspk's own items (`--frac 0.32`, share 0.34; memory: trains 3.5 GB, -hp 11 GB;
  reads 5 GB);
- 2 compares, the pair, the re-call and the verdict.

No build: zspkb reads zspk's carves, already on the host, and computes its inputs in the batch.
