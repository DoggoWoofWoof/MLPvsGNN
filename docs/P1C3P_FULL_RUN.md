# P1c3p's full run: zrckp, the pointwise KT head, on all six splits

Declared 14 October 2026 at 19:40. P1c3p's screen is still running: its two fits are training, and none of its reads,
comparisons or numbers exists yet. P1c3's full run is filed ADOPT (`outputs/p1c3/full-verdict.json`, 19:23).

Nothing below depends on P1c3's full-run numbers:
- the protocol is P1c3's full run (docs/P1C3_FULL_RUN.md), carried to one arm;
- the rule is P1c3p's section 5.

These are development numbers. The paper's numbers come from one declared confirmation run.

## 1. Question

P1c3p (docs/P1C3P_POINTWISE_HEAD.md) found that zrck's head breaks the MLP's 8 October rule:
- One of its inputs is the row's score z-scored over its question's rows, so a row's final score depends on the other
  rows' scores. zrck serves no claim, whatever its full run says.
- zrckp is the fix: zrc's model plus a head that reads only the row's own eight KT columns (161 parameters, no score
  input).

P1c3p's section 5: ADOPT_FOR_FULL_RUN, together with P1c3's full run ADOPT, leads to "zrckp's full run in its own
declared file (six splits, grade-nullx)". This is that file. It is gated on both verdicts (section 5 below).

Does the pointwise KT head lift zrc's reads on every split, with nothing losing? The splits are J5 (in-domain on all
five training datasets) and each leave-one-out fit, read on its held-out dataset.

## 2. Arm and fits, all on the card

All commands are `outputs/mp_unified/zp1c3pf.py` (new; frozen once run). It imports `zp1c3p.py` and `zp1c3f.py`
unchanged.

**The arm is P1c3p's, unchanged:**
- **zrckp**: zrc's model (zrm.ZRM) plus the pointwise KT head, 8 -> 16 (GELU) -> 1, added to the score. Its first
  layer is drawn from seed + 3301 and its last layer starts at zero, so at the start it is zrc bit for bit.
- This is the MLP: no message passing, and no input from another row's score.
- Its carve, roots, settings, seed (0), batches, candidate (p@swa) and device (the card) are P1c3p's. It trains on
  P1c2's pools: `--cache-root outputs/p1c2/cache`.

**The fits:**
- The screen's fits are its L-musique and L-hotpotqa fits: `scr-p1c3p-zrckp` and `scr-p1c3p-zrckp-hp`.
- Four more: J5, L-2wiki, L-squad and L-metaqa, into `outputs/full_zrckp/fits`.
  `zp1c3pf.py train --arm zrckp --split S --name S --out-root outputs/full_zrckp/fits --device cuda --host
  --cache-root outputs/p1c2/cache`.
- `zp1c3pf.py train` runs zp1c3p.py's train steps (the cache root, the card, P1c2's gate, the KT records of the split's
  fit carves, the fit on P1c2's roots, P1c3p's stamp) on these four splits. Nothing else changes.

**Every input these fits read already exists.** They are the inputs zrck's full fits read: P1c2's pools, caches,
chains and links, and P1c3's KT inputs for the five fit carves and the six s1eval carves. There is no new pool, row,
column, graph, edge or input.

## 3. Reads and comparisons

- Each fit is read on the six s1eval carves of P1c2's pools. `zp1c3pf.py read` is zp1c3p.py's read.
- **Each read is compared, question by question, with zrc's card fit of its split on today's pools.** That comparison
  decides. These are the bases zrck's full run was decided against:

  | L-musique | L-hotpotqa | the other four splits (S) |
  | --- | --- | --- |
  | `scr-zrct` | `scr-zrct-hp` | `outputs/full_zrct/fits/S` |

- **Reported beside, deciding nothing:** zrck's fit of the split (`scr-p1c3-zrck`, `scr-p1c3-zrck-hp`,
  `outputs/full_zrck/fits/S`), giving zrckp minus zrck, and step 1's fit.
- Every compared read covers the same questions and the same gold totals. A gold the pool loses counts as a miss.
- **The comparison** is lean_screen's: the R@5 difference with a 2,000-resample question bootstrap, called GAIN, LOSS
  or WITHIN with the floor 0.0075. FC@5 and hit@1 are reported beside it.
- There are 36 reads. The primary reads are step 1's eleven: J5 on all six datasets (five in-domain, webqsp
  zero-shot), and each leave-out fit on its held-out dataset.

## 4. Verdict

**The grade:**
- `zp1c3pf.py grade` files relz's grade against zrc's card fits.
- The screen's comparisons are reused for L-musique and L-hotpotqa: `outputs/screen/scr-p1c3p-zrckp.json` and
  `outputs/screen/scr-p1c3p-zrckp-hp.json`.

**The re-grade decides each read's call.**
- `nullx.py regrade` re-grades all 36 reads under the seed null over every split (the twelve null files of P1c3's full
  run), into `outputs/full_zrckp/grade-nullx`.
- Each read's base R@5 is its base fit's, from that fit's comparison with step 1's: `outputs/zbase2/base-zrc-S.json`,
  as in zrck's full run.

**The verdict** (`zp1c3pf.py verdict`, into `outputs/p1c3p/full-verdict`):
- **ADOPT:** the re-grade has no LOSS on any of its 36 reads, and at least one primary GAIN.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE** (exit 1): the re-grade is missing, has the wrong arm, has other base comparisons, is itself
  INCOMPLETE, or does not have the 36 reads with 11 primary.
- This is P1c3's full-run rule for one arm. The re-grade's own verdict is reported beside and decides nothing.

**What ADOPT means:**
- The lean pair becomes (zrckp, zspk), both on KIND_T's pools with the KT input.
- zrckp's head reads only the row's own eight columns. zspk's head keeps P1c3's score input: zspk is the GNN, whose
  rows read other rows by design. Both read the same eight columns.
- Before any claim uses the pair, its serving cost is timed cold in its own declared file. P4c (docs/P4C_KT_SERVING.md)
  times zrck's 177-parameter head, an upper bound for zrckp's 161.
- The universal pair (twin0, gnn0) is not changed by this run.

**What NOT_ADOPTED means:**
- The lean MLP stays zrc. Moving only the GNN would compare the two models on different pools and inputs (P1c3's full
  run, section 4), so the lean pair stays (zrc, zsp) on today's pools.
- zspk's part of P1c3's ADOPT stays filed and moves nothing alone.
- This run's record says, read by read, where the pointwise head moved R@5, and P3's ledger continues from there.

**Every fit runs to its end.** A LOSS at the floor 0.0075 may turn WITHIN under the null, so nothing is stopped early.

## 5. Gates and order of work

**The fits start now,** on the card's free slot, before P1c3p's screen verdict:
- A fit's training reads nothing of the screen and cannot change it.
- The screen's reads stay ahead of these fits in the feeder's order, so the screen is not delayed.
- This uses the card while the screen finishes (two of its three slots are busy now).

**`p1c3pf-gate`** (`zp1c3pf.py gate`) waits for P1c3p's verdict (`p1c3p-verdict`).
- It reads `outputs/p1c3p/screen-verdict.json` and `outputs/p1c3/full-verdict.json`, and writes
  `outputs/p1c3p/full-gate.{json,md}` with both records' sha256.
- It exits 0 only when P1c3p's screen is ADOPT_FOR_FULL_RUN and P1c3's full run is ADOPT.

**Every read, comparison, grade and re-grade waits on the gate.**
- `read` and `compare` also refuse in code unless both records say so.
- So if the screen is not ADOPT_FOR_FULL_RUN, the four fits are never read. Their folders go on the next clean-up
  list.

**Order and caps:**
- The fits go in this order: J5 first (it trains on all five datasets, so it is the longest), then L-2wiki, L-squad
  and L-metaqa.
- Every fit and read runs under cuda_alloc (frac 0.30, share 0.32), so three share the card.
- Host memory uses zrck's full-run sizes, since zrckp is zrck's model less one head input: each fit reserves 5 GB
  (6 GB on J5), and each read 5 GB.

**ETA:**
- J5 about 50 minutes, from about 19:45.
- L-2wiki and L-squad start after the screen's two reads (about 19:55). L-metaqa starts when J5 ends (about 20:35).
- The last fit ends about 21:15. The reads, comparisons, grade, re-grade and verdict follow: about 21:30.

## 6. What this does not do

- It adds no new graph, pool, row, column, text, encoder or model. The encoder and the substrate embeddings stay frozen.
- It adds no new seed, setting or hyperparameter.
- It writes nothing into zrc's, zrck's, step 1's, P1c2's or the screen's fit folders. This run's fits are a new
  folder: `outputs/full_zrckp/fits`.
- webqsp never trains. Test splits are never read.
- It does not read speed.

## Self-test

`zp1c3pf.py --selftest` passes:
- the arm, the four new splits, and zrck's full run's base fits and base comparisons;
- the gate (five cases) and three refusals: a record of another stage, of another declared file, or missing;
- reads and comparisons refused until the gate opens, and a screen fit refused as this run's;
- train refusing another arm, a screen split, another name or folder, and leaving the splits unchanged;
- the grade against zrc's card fits (three cases), and INCOMPLETE when zrck's fits decide;
- the verdict (four cases) and seven INCOMPLETEs, with zrckp minus zrck beside.

## Results

To be filed.
