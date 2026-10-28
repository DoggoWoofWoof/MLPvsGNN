# PSMH: the hard gap (zrckph) on Kaggle's T4

Declared 28 October 2026, after PSMK's verdict (46398e2) and before any zrckph fit. PSMK's compares show WQK's seed-0
zrckp R@5 on the twelve reads. zrckph's form was fixed before any number by PSM's section 6
([PSM_POOL_SOFTMAX_INPUT.md](PSM_POOL_SOFTMAX_INPUT.md)). Its device, bases, floors and rule are PSMK's (sections 2 and 3
of [PSMK_PSM_ON_T4.md](PSMK_PSM_ON_T4.md)), which PSMK's section 3 lets a next form declare again. This file adds only
zrckph's next forms (section 5). Code: `outputs/mp_unified/zpsmh.py` and `outputs/mp_unified/zpsmh_session.py`.

## 1. Why this, now

- **PSMK is NOT_ADOPTED with no GAIN** (46398e2): 0 GAIN and 7 LOSS under WQK's floors on zrckp. For that outcome PSM's
  section 6 fixed the next form before any number: "zrckph: the hard gap, z_i minus the pool's largest z, floored at
  -10".
- **A T4 is free.** PSMK's session has ended, so mpr-kg is free. The host is still offline.
- **It costs one session**, about 35 minutes on two T4s (PSMK's took 2,087 s).

## 2. What changes, and what does not

| | PSMK | PSMH |
|---|---|---|
| arm | zrckpg: each fixed column's log-softmax of zret's z over the question's rows, at temperature 1, floored at -10 | zrckph: each fixed column's z minus the largest z of its column over the question's rows, floored at -10 |
| fits | `scr-psmk-zrckpg`, `scr-psmk-zrckpg-hp` | `scr-psmh-zrckph`, `scr-psmh-zrckph-hp` |
| stamp key | `psmk` | `psmh` |
| record | `outputs/psmk/screen-verdict` | `outputs/psmh/screen-verdict` |

Everything else is PSMK's, unchanged:
- the device: a T4 in a Kaggle kernel on KG's image;
- WQK's roots, with WQK's mixed KT input;
- the reads: the six s1eval carves at p@swa, with `--rows-cap 65536`;
- the bases: `scr-wqk-zrckp-s0` and `scr-wqk-zrckp-s0-hp`;
- the floors: `outputs/wqk/floors-zrckp.json`;
- the verdict code: zpsm's;
- the code: zpsmk.py and zpsmk_session.py, imported unchanged and run under PSMH's names by zpsmh.py and
  zpsmh_session.py. zpsm.py, zpsmk.py and zpsmk_session.py do not change.

## 3. The arm

For each fixed block b (every block but SEMB), each of its columns c, and each row i of question q:

    g_ic = max(z_ic - max_{j in q} z_jc, -10) * keep_qb

- **z** is zret's z-score of the column (`lean_screen3.seg_zscore_ref`), against the question's retrieved rows, as in
  PSM.
- **The weights.** The gaps enter the first layer through W_g: hidden x gap columns, zero at the start, no bias, drawn
  from no generator. So at the start zrckph is zrckp bit for bit, as zrckpg is. A fit has 86 gap columns and 11,008
  gap weights, as PSMK's fits had.
- **How it differs from PSM's gap.**
  - zrckpg's gap is the log-softmax, z_i - log sum_j exp(z_j). Times its temperature T, the soft gap goes to the hard
    gap as T goes to 0.
  - In zrckph the pool's best row of each column has gap 0, whatever the pool. In zrckpg the best row's gap falls as
    more rows come close to it, and a constant column gives -log n to every row. In zrckph a constant column gives 0.
- **No message passing.** As with PSM's gap, the input aggregates over the question's rows, but only over the cache's
  fixed columns, as zret's z-scores do. No learned value and no row's score enters it, and it is computed before any
  row is scored. As in PSM, SEMB (whose columns the model computes) and the KT input take no gap. A block whose
  columns are not the batch's own is refused.
- **The segment max** is the op lean_gpu's loss already runs under deterministic algorithms (`scatter_reduce` with
  `amax`).

## 4. The rule (PSMK's section 3, with PSMH's names)

- **The re-call.** Each of the 12 reads (2 splits x 6 datasets) is called again under WQK's floors on zrckp
  (`outputs/wqk/floors-zrckp.json`, status OK, WQK's), by screen_recall.recall on the filed call and the R@5
  difference. A read's base R@5 must equal the floors record's seed-0 R@5 for that read. Otherwise the verdict is
  INCOMPLETE.
- **The verdict**, by zpsm.py's own verdict code with PSMH's names:
  - **ADOPT_FOR_FULL_RUN** when no read is LOSS and at least one is GAIN;
  - **NOT_ADOPTED** otherwise;
  - **INCOMPLETE** when a comparison or the floors are missing, are not WQK's or PSMH's, or do not hold 12 reads.
- **Beside, deciding nothing:** the FC@5 and hit@1 differences per read, and each fit's train and read seconds.

## 5. What follows (fixed now, before any zrckph number)

Each next form gets its own declared file before its fit. Each may declare the T4 again, with WQK's bases and floors.
- **ADOPT_FOR_FULL_RUN → zspkh:** the hard gap on the GNN (zspk), so that F1's controlled pair keeps one input set.
  Then the pair's full run under WQ.
- **NOT_ADOPTED with a GAIN and a LOSS → zrckph1:** the hard gap divided by 10, so in [-1, 0].
- **NOT_ADOPTED with no GAIN → zrckpg1:** PSM's soft gap divided by 10, so in [-1, 0].
- **INCOMPLETE:** the missing piece reruns. Nothing is decided.

**Why the scale forms.** Both gaps reach -10, while the z-scores beside them have unit spread over the retrieved rows.
lean_gpu trains with Adam, whose step on a weight does not grow with its gradient. So a step on W_g can move a
pre-activation up to ten times as far as a step on a z-score's weight. The divided forms keep each gap's order and
change only its scale. They test whether PSMK's losses come from the gaps' scale or from the input itself.

## 6. How it runs

**One session**, a new version of whichever of KG's two kernels is free (mpr-kg or mpr-kg-g6b2). That keeps KG's
pinned image.
- **Datasets:** KG's nine, WQK's three groups (mpr-wqk-1, -2, -3) and the code bundle. They are checked and linked by
  WQK's session code, unchanged.
- **Before anything runs:** WQK's verdict (`outputs/wqk/base-verdict.json`) must be BASE_READY under WQK's
  declaration, and the clone must be its commit.
- **Before any fit:** WQK's overlay (W9) and gate (W10), each in its own process. An overlay or gate that does not exit
  0 stops the session.
- **The queues:**
  - GPU 0: `scr-psmh-zrckph-hp`;
  - GPU 1: `scr-psmh-zrckph`.

  No fit starts after 8 hours. Where a fit runs decides nothing.
- **Each fit**, in its own process: `zpsmh.py fit --name N --rows-cap 65536`. This is zpsmk's fit under PSMH's names,
  with zpsm's arm swapped for zrckph. It refuses a name not PSMH's, a device that is not a T4 on KG's image, another
  rows cap, a gate that did not pass, and a WQK verdict that is not BASE_READY. Its timing record keeps zpsmk's file
  name, `psmk_timing.json`, with PSMH's declaration inside. Its messages keep zpsmk's wording, with PSMH's names in
  them (for example "is not one of PSMK's fits (scr-psmh-zrckph, …)").
- **Time and memory.** As PSMK's: about 35 minutes on two T4s, and about 4.6 GB of RAM per fit.

**On the laptop, after the session:**

    python outputs/mp_unified/zpsmh.py take --from <the kernel's output folder>
    python outputs/mp_unified/zpsmh.py compare-all
    python outputs/mp_unified/zpsmh.py verdict

These are PSMK's take, compare-all and verdict (its section 4), with PSMH's names. Outputs:
- `outputs/screen/scr-psmh-zrckph.json` and `scr-psmh-zrckph-hp.json`;
- `outputs/psmh/screen-verdict.json` and `.md`;
- the session's gate and overlay under `outputs/psmh/sessions/<the gate's sha256[:12]>/`.

**The stub** (`zpsmh_session.py stub`) refuses what PSMK's refuses. It also refuses a HEAD that does not track
zpsmh.py, zpsmh_session.py and this file.

**What is committed:**
- Committed: this file, the code, the two compares and the verdict.
- Never committed: the fits' folders, weights, reads and logs, session.json, the datasets' manifests, and
  `outputs/psmh/sessions`.

## 7. Self-tests, passed before this commit

`python outputs/mp_unified/zpsmh.py --selftest` and `python outputs/mp_unified/zpsmh_session.py --selftest` pass on
the laptop. They check:
1. **The names.**
   - zrckph's fits are on WQ's two splits, distinct from PSM's, PSMK's, WQK's and WQ's.
   - The bases are WQK's seed-0 zrckp fits, and the floors are WQK's.
   - Under PSMH's names, zpsmk's names are PSMH's, and zpsm's arm, input and next forms are zrckph's. zpsmk's own
     check of zpsm's arm passes.
   - All are restored after, on a failure too.
2. **The arm.**
   - It is zrckpg's class with the hard gap. zrckpg's, zrckp's, zrc's and zspk's arms are unchanged.
   - From the same seed, zrckph's state is zrckpg's, and its forward is zrckp's bit for bit: in eval, and in training
     with the same dropout draws.
3. **The gaps**, against a float64 reference:
   - each pool's best row is 0 in every column;
   - a constant column is 0, and a dropped block is zero;
   - a gap far below the best is floored at -10;
   - the soft gap times its temperature comes within 0.01 of the hard gap at T = 0.001.

   No learned value, SEMB input or KT column moves the gaps. One question's columns move only its own gaps, and a row
   permutation permutes them. A learned block is refused. Two toy fits move W_g and are bit-identical.
4. **The guards, stamps, take, compare-all and verdict**, as PSMK's self-test checks them, with PSMH's names:
   - a PSMK, PSM or WQK name is refused, and so is zrckpg under PSMH's prefix;
   - the stamps carry PSMH's block, not PSMK's, and the psm block names PSMH and the hard gap;
   - take files the gate under `outputs/psmh/sessions`;
   - the verdict's eleven cases: ADOPT_FOR_FULL_RUN (zspkh); NOT_ADOPTED with a GAIN and a LOSS (zrckph1) and with no
     GAIN (zrckpg1); INCOMPLETE for eleven reads, a compare not PSMH's (PSMK's among them), a compare made on another
     device, a fit and its base on two devices, floors not WQK's, a base R@5 off the floors, and a missing compare.
5. **The session**, on a toy repository with no zpsmk.py:
   - the overlay and gate run first, then `zpsmh.py fit --name N --rows-cap 65536`;
   - a failing overlay or gate stops it, and the names and the rows cap are checked;
   - the stub runs zpsmh_session.py, and refuses a HEAD that tracks PSMK's files but not PSMH's.
6. **Nothing is written** under PSMH's or PSMK's real folders.

The text encoder stays frozen. Test splits are never read. There is no LLM, no topic-entity help and no new graph.
webqsp trains under WQ's ruling only.

## 8. Results

None yet.
