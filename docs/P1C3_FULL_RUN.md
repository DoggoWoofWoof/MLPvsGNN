# P1c3's full run: the KT input on KIND_T's pools, both models, all six splits

Declared 14 October 2026 at 17:09, while P1c3's screen is still running. Its verdict is not in, and no fit, read or
number of this run exists yet.

When this was written, three of the screen's four comparisons had landed, and one arm's re-call (zspk's) had been
written. Of these, only one comparison had been read (zrck on L-hotpotqa). Nothing below depends on any of them:
- the protocol is round 33's full run (docs/FULL_ROUND33.md) and P1c3's section 5;
- the pair rule is P1c3's screen rule, carried from twelve reads per arm to thirty-six.

These are development numbers. The paper's numbers come from one declared confirmation run.

## 1. Question

P1c3 (docs/P1C3_KT_INPUT.md) adds one input to both models: each pool row's place in KIND_T's order and in the walks
KIND_T is built from, whether the row is in KIND_T's fill, and the question's kind. Both models get the same eight
columns, computed when the pool is built, before any row is scored, from no gold. Both models train on KIND_T's pools
(P1c2's).

Its section 5: "ADOPT_FOR_FULL_RUN: the full run, six splits per arm under grade-nullx, in its own file". This is that
file. It is gated on the screen's verdict (section 5 below).

Does the KT input on KIND_T's pools lift both models' reads on every split, with nothing losing? The splits are J5
(in-domain on all five training datasets) and each leave-one-out fit, read on its held-out dataset.

## 2. Arms and fits, all on the card

All commands are `outputs/mp_unified/zp1c3f.py` (new; frozen once run). It imports `zp1c3.py` unchanged.

**The arms are P1c3's, unchanged:**
- **zrck**: zrc's model (zrm.ZRM) plus the KT head, on zrc's chain carve with the KT input beside it. This is the MLP:
  no message passing.
- **zspk**: zsp's model (zprop.ZProp) plus the KT head, added before zsp's propagation reads the score. This is the
  GNN.
- Their head, carves, roots, settings, seed (0), batches, candidate (p@swa) and device (the card) are all P1c3's.
- Both train on P1c2's pools: `--cache-root outputs/p1c2/cache`, with P1c2's chains and links.

**The fits:**
- The screen's fits are each arm's L-musique and L-hotpotqa fits: `scr-p1c3-zrck`, `scr-p1c3-zrck-hp`,
  `scr-p1c3-zspk` and `scr-p1c3-zspk-hp`.
- Each arm gets four more fits: L-2wiki, L-squad, J5 and L-metaqa, eight in all.
  `zp1c3f.py train --arm A --split S --name S --out-root outputs/full_A/fits --device cuda --host --cache-root
  outputs/p1c2/cache`.
- `zp1c3f.py train` is zp1c3.py's train with these four splits in place of the screen's two. Nothing else changes.

**Every input these fits read already exists:**
- P1c2's pools, caches, chains and links for the fit carves of all five training datasets. They were built for the
  screen's two splits, which together train on all five.
- P1c3's eleven KT inputs: the five fit carves and the six s1eval carves.
- So there is no new pool, row, column, graph, edge or input.

## 3. Reads and comparisons

- Each fit is read on the six s1eval carves of P1c2's pools. `zp1c3f.py read` is zp1c3.py's read.
- **Each read is compared, question by question, with the same family's card fit of its split on today's pools.**
  That comparison decides.

  | arm | L-musique | L-hotpotqa | the other four splits (S) |
  | --- | --- | --- | --- |
  | zrck | `scr-zrct` | `scr-zrct-hp` | `outputs/full_zrct/fits/S` (round 33's bases) |
  | zspk | `scr-zspg` | `scr-zspg-hp` | `outputs/full_zspg/fits/S` |

  Step 1's fit of the split is reported beside each.
- Every compared read covers the same questions and the same gold totals. R@5 is taken over a question's golds, so a
  gold the pool loses counts as a miss.
- **The comparison** is lean_screen's: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS
  or WITHIN with the floor 0.0075. FC@5 and hit@1 are reported beside it.
- Each arm has 36 reads. The primary reads are step 1's eleven: J5 on all six datasets (five in-domain, webqsp
  zero-shot), and each leave-out fit on its held-out dataset.

## 4. Verdict: one for the pair

**Each arm's grade:**
- `zp1c3f.py grade --arm A` files relz's grade against A's base fits.
- The screen's comparisons are reused for L-musique and L-hotpotqa: `outputs/screen/scr-p1c3-A.json` and
  `outputs/screen/scr-p1c3-A-hp.json`.

**The re-grade decides each read's call.**
- `nullx.py regrade` re-grades all 36 reads under the seed null over every split into `outputs/full_A/grade-nullx`.
- Each read's base R@5 is its base fit's, from that fit's comparison with step 1's:
  - zrc's: `outputs/zbase2/base-zrc-S.json`;
  - zsp's: `outputs/zg1/base-zspg-S.json` on L-musique and L-hotpotqa, and `outputs/full_zspg/base-zspg-S.json` on the
    other four.

**The pair's verdict** (`zp1c3f.py verdict`, into `outputs/p1c3/full-verdict`):
- **ADOPT:** neither arm's re-grade has a LOSS on any of its 36 reads, and the two arms together have at least one
  primary GAIN.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE** (exit 1): a re-grade is missing, has the wrong arm, has other base comparisons, or does not have the
  36 reads.
- Each arm's own re-grade verdict uses round 33's rule (a primary GAIN and no LOSS). It is reported beside and decides
  nothing.

**Why one verdict for the pair:**
- zrc and zsp are the lean pair: the universal-lean family of [C9_UNIVERSAL_PAIR.md](C9_UNIVERSAL_PAIR.md). Within a
  family both models read the same pools, carves and seeds, so KIND_T's pools move both models or neither.
- Moving one model alone would compare the MLP and the GNN on different pools and inputs.
- Moving both, when one gains and neither loses, leaves each model at least where it was.
- This is P1c3's screen rule (two arms together, no LOSS, a GAIN), carried to the full run. Here the GAIN must be a
  primary read, as in every full run.

**What ADOPT means:**
- zrck and zspk, on KIND_T's pools with the KT input, become the lean pair: the claims' lean models.
- Before any claim uses them, their serving cost is timed cold in its own declared file. That is P1c3's section 5:
  every walk the KT input reads, on P4b's faster exact walks.
- The universal pair (twin0, gnn0) is not changed by this run.

**Corrected 14 October at 17:27, in wording only.** This section first had the two families' names swapped. It
called zrck and zspk the universal pair, twin0 and gnn0 the lean pair, and gave as the reason for one verdict that both
models of the universal pair read the same inputs. zrc and zsp are the lean pair, and twin0 and gnn0 the universal pair
([C9_UNIVERSAL_PAIR.md](C9_UNIVERSAL_PAIR.md), [F1_ONE_INPUT_SET.md](F1_ONE_INPUT_SET.md)). The reason is the family's
shared pools. No rule, base, split, command or file changes.

**What NOT_ADOPTED means:**
- Today's pair stays.
- This run's record says, read by read, where the KT input moved R@5 on all six splits, and P3's ledger continues from
  there.

**Every fit runs to its end.** A LOSS at the floor 0.0075 may turn WITHIN under the null, so nothing is stopped early.

## 5. Gates and order of work

**The fits start now,** on the card's free slots, before the screen's verdict:
- A fit's training reads nothing of the screen and cannot change it.
- The screen's last read stays ahead of these fits in the feeder's order, so it is not delayed.
- This uses the card while the screen finishes (one of its three slots is busy now).

**`p1c3f-gate`** (`zp1c3f.py gate`) waits for both of the screen's re-calls.
- It applies P1c3's rule to them and writes `outputs/p1c3/screen-verdict.{json,md}`. That record is also the
  screen's verdict for P1c3's own results section.
- It exits 0 only on ADOPT_FOR_FULL_RUN.

**Every read, comparison, grade and re-grade waits on the gate.**
- `read` and `compare` also refuse in code unless the gate's record says ADOPT_FOR_FULL_RUN.
- So if the screen is NOT_ADOPTED, the eight fits are never read. Their folders go on the next clean-up list.

**Order and caps:**
- The fits go in this order: J5 first (it trains on all five datasets, so it is the longest), then L-2wiki, L-squad
  and L-metaqa, alternating the arms, zspk first.
- Every fit and read runs under cuda_alloc (frac 0.30, share 0.32), so three share the card.
- Host memory is sized from the screen's measured peaks: 2.1-4.0 GB per fit and about 4 GB per read. So each fit
  reserves 5 GB (6 GB on J5), and each read 5 GB.

**ETA:**
- The screen's fits took 37-43 minutes with three on the card. J5's should take about 50.
- Eight fits run, two at a time until the screen's last read is done, then three at a time. The last fit ends about
  19:25.
- The reads, comparisons, grades, re-grades and the verdict follow: about 19:40.

## 6. What this does not do

- It adds no new graph, pool, row, column, text, encoder or model. The encoder and the substrate embeddings stay frozen.
- It adds no new seed, setting or hyperparameter.
- It writes nothing into zrc's, zsp's, step 1's, P1c2's or the screen's fit folders. This run's fits are new folders:
  `outputs/full_zrck/fits` and `outputs/full_zspk/fits`.
- webqsp never trains. Test splits are never read.
- It does not read speed. The KT input's serving cost is timed in its own file before any claim uses a KT model.

## Self-test

`zp1c3f.py --selftest` passes:
- the four new splits, and each arm's base fits and base comparisons;
- the gate on P1c3's rule (five cases) and five refusals of a re-call that is not P1c3's or lacks a read;
- reads and comparisons refused until ADOPT_FOR_FULL_RUN;
- train refusing the screen's splits, another arm or another folder, and leaving zp1c3's splits restored;
- each arm graded against its own family's bases, and graded against the other family's bases it is INCOMPLETE;
- the pair's verdict (five cases) and six INCOMPLETEs.

## Results

Filed 14 October 2026 at 19:41. The verdict was written on the host at 19:23. Records:
- `outputs/p1c3/full-verdict.{json,md}`;
- `outputs/full_zrck/` and `outputs/full_zspk/`: each arm's grade, its re-grade under the null, and its four
  comparisons.

**Verdict: ADOPT.** Neither arm has a LOSS on any of its 36 reads under the seed null. Each arm has five primary
GAINs, on the same five reads: J5 metaqa, J5 2wiki, J5 webqsp, L-metaqa metaqa and L-2wiki 2wiki.

| arm | primary GAIN | GAIN in all | LOSS | filed grade (floor 0.0075) -> re-grade |
| --- | --- | --- | --- | --- |
| zrck (MLP) | 5 of 11 | 18 of 36 | 0 | NOT_ADOPTED -> ADOPT |
| zspk (GNN) | 5 of 11 | 16 of 36 | 0 | NOT_ADOPTED -> ADOPT |

**Why the filed grades said NOT_ADOPTED.** At the flat floor 0.0075, each arm's L-musique fit had one LOSS, on its
held-out musique read: zrck -0.0176 and zspk -0.0100. Under the null that read's floor is 0.0720, the widest of the
36, so both calls are WITHIN. Section 4 fixed the re-grade as the decider before any number.

**R@5, each arm minus its family's card fit on today's pools.** Bold: primary reads. zs: zero-shot. A changed call
shows filed -> re-call.

zrck (outputs/full_zrck/grade-nullx.md):

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| J5 | **+0.0196 GAIN** | **+0.0001** | **+0.0019** | **-0.0012** | **+0.0573 GAIN** | **+0.0529 GAIN zs** |
| L-metaqa | **+0.0619 GAIN zs** | +0.0003 | +0.0042 | +0.0063 | +0.0662 GAIN | +0.0212 GAIN zs |
| L-squad | +0.0216 GAIN | **-0.0028 zs** | +0.0061 | +0.0034 | +0.0623 GAIN | +0.0721 GAIN zs |
| L-musique | +0.0225 GAIN | +0.0013 | **-0.0176 LOSS -> WITHIN zs** | +0.0022 | +0.0632 GAIN | +0.0372 GAIN zs |
| L-hotpotqa | +0.0259 GAIN | +0.0009 | +0.0050 | **+0.0113 GAIN -> WITHIN zs** | +0.0397 GAIN | +0.0820 GAIN zs |
| L-2wiki | +0.0215 GAIN | -0.0019 | +0.0039 | +0.0045 | **+0.0639 GAIN zs** | +0.0548 GAIN zs |

zspk (outputs/full_zspk/grade-nullx.md):

| split | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| J5 | **+0.0211 GAIN** | **-0.0009** | **+0.0110 GAIN -> WITHIN** | **-0.0011** | **+0.0586 GAIN** | **+0.0331 GAIN zs** |
| L-metaqa | **+0.0422 GAIN zs** | -0.0029 | +0.0073 | +0.0010 | +0.0563 GAIN | +0.0021 zs |
| L-squad | +0.0255 GAIN | **+0.0015 zs** | +0.0075 | +0.0028 | +0.0533 GAIN | +0.0263 GAIN zs |
| L-musique | +0.0215 GAIN | -0.0011 | **-0.0100 LOSS -> WITHIN zs** | +0.0015 | +0.0659 GAIN | -0.0017 zs |
| L-hotpotqa | +0.0233 GAIN | +0.0026 | +0.0107 GAIN | **+0.0064 zs** | +0.0589 GAIN | +0.0418 GAIN zs |
| L-2wiki | +0.0247 GAIN | +0.0004 | +0.0070 | +0.0032 | **+0.0663 GAIN zs** | +0.0351 GAIN -> WITHIN zs |

Unmarked cells are WITHIN.

**Where the KT input moves R@5:**
- 2wiki: up on every split for both models, +0.040 to +0.066, including its own held-out read. FC@5 rises up to
  +0.13.
- metaqa: up on every split, zrck +0.020 to +0.062 and zspk +0.021 to +0.042. The largest gain is the held-out read
  (L-metaqa).
- webqsp, always zero-shot: zrck up on all six splits (+0.021 to +0.082); zspk on four (+0.026 to +0.042), with
  L-metaqa and L-musique level.
- squad and hotpotqa: within +-0.007 everywhere except zrck's held-out hotpotqa read (+0.0113, WITHIN under its floor
  0.0200).
- musique: down on its held-out read (zrck -0.018, zspk -0.010; both WITHIN), up +0.002 to +0.011 on the other
  splits.

**Beside, deciding nothing (from each arm's grade.json):**
- hit@1 falls on hotpotqa's held-out read for both models (zrck -0.042, zspk -0.043), as it did on the screen. zspk's
  J5 hotpotqa hit@1 also falls (-0.016).
- hit@1 rises on musique on all six splits for both models (zrck +0.012 to +0.055, zspk +0.019 to +0.044). That
  includes the held-out read (zrck +0.053, zspk +0.031), where R@5 falls.

**What follows:**
- zrck serves no claim. Its head reads the row's score z-scored over its question's rows, which breaks the MLP's
  8 October rule (docs/P1C3P_POINTWISE_HEAD.md). So the lean pair does not move on this verdict alone, as section 4's
  one verdict for the family requires.
- zrckp, the pointwise head, decides the lean MLP: its screen (P1c3p) and its full run
  ([P1C3P_FULL_RUN.md](P1C3P_FULL_RUN.md), declared 19:40). ADOPT there makes the lean pair (zrckp, zspk). Anything
  else keeps (zrc, zsp) on today's pools, and zspk's part of this ADOPT moves nothing alone.
- Before any claim uses a KT model, its serving cost is timed cold (docs/P4C_KT_SERVING.md).
