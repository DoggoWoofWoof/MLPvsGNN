# PSMS: the gaps at the z-scores' scale (zrckpg1 and zrckph1) on Kaggle's T4

Declared 28 October 2026, after PSMH's verdict (ad371da) and before any zrckpg1 or zrckph1 fit. PSMK's and PSMH's
compares show WQK's seed-0 zrckp R@5 on the twelve reads, and their own. zrckpg1's form was fixed before any number by
PSMH's section 5 ([PSMH_HARD_GAP_ON_T4.md](PSMH_HARD_GAP_ON_T4.md)); zrckph1 is the other divided form that section
names. Their device, bases, floors and rule are PSMK's (sections 2 and 3 of [PSMK_PSM_ON_T4.md](PSMK_PSM_ON_T4.md)),
which PSMH's section 5 lets a next form declare again. This file adds the second arm, one session per arm, and the next
forms (section 5). Code: `outputs/mp_unified/zpsms.py` and `outputs/mp_unified/zpsms_session.py`.

## 1. Why this, now

- **PSMH is NOT_ADOPTED with no GAIN** (ad371da): 0 GAIN and 2 LOSS (webqsp on both splits) under WQK's floors on
  zrckp. For that outcome PSMH's section 5 fixed the next form before any number: "zrckpg1: PSM's soft gap divided by
  10, so in [-1, 0]."
- **Why zrckph1 beside it.** PSMH's section 5 gives one reason for both divided forms: they "keep each gap's order and
  change only its scale. They test whether PSMK's losses come from the gaps' scale or from the input itself." zrckpg1
  asks it of the soft gap and zrckph1 of the hard gap; one alone answers half of it. PSMH's section 5 named zrckph1 for
  a GAIN and a LOSS. Here it is screened as its own arm, with its own verdict, while zrckpg1 runs.
- **A T4 each.** mpr-kg is free after PSMH. mpr-kg-g6b2 has been idle since G6b2's verdict, and G6RW needs it only once
  its files, typed links and encoder are in place. Each session costs about 37 minutes (PSMH's took 2,230 s).

## 2. What changes, and what does not

| | PSMH | PSMS |
|---|---|---|
| arms | zrckph | zrckpg1: PSM's soft gap divided by 10; zrckph1: PSMH's hard gap divided by 10 |
| fits | `scr-psmh-zrckph`, `scr-psmh-zrckph-hp` | `scr-psms-zrckpg1`, `scr-psms-zrckpg1-hp`, `scr-psms-zrckph1`, `scr-psms-zrckph1-hp` |
| sessions | one | one per arm |
| stamp key | `psmh` | `psms` |
| records | `outputs/psmh/screen-verdict` | `outputs/psms/zrckpg1-verdict`, `outputs/psms/zrckph1-verdict` |

Everything else is PSMK's, unchanged:
- the device: a T4 in a Kaggle kernel on KG's image;
- WQK's roots, with WQK's mixed KT input;
- the reads: the six s1eval carves at p@swa, with `--rows-cap 65536`;
- the bases: `scr-wqk-zrckp-s0` and `scr-wqk-zrckp-s0-hp`;
- the floors: `outputs/wqk/floors-zrckp.json`;
- the verdict code: zpsm's;
- the code: zpsmk.py and zpsmk_session.py, imported unchanged and run under each arm's PSMS names by zpsms.py and
  zpsms_session.py. zpsm.py, zpsmk.py, zpsmk_session.py, zpsmh.py and zpsmh_session.py do not change.

## 3. The arms

For each fixed block b (every block but SEMB), each of its columns c, and each row i of question q:

    zrckpg1:  g_ic = max(z_ic - log sum_{j in q} exp(z_jc), -10) / 10 * keep_qb
    zrckph1:  g_ic = max(z_ic - max_{j in q} z_jc, -10) / 10 * keep_qb

- **z** is zret's z-score of the column (`lean_screen3.seg_zscore_ref`), against the question's retrieved rows, as in
  PSM and PSMH.
- **The division.** Each gap is computed as before, by zpsm's code (zrckpg1) or zpsmh's (zrckph1), and then divided by
  10. So each lies in [-1, 0], keeps its order, and is exactly its undivided arm's gap over 10.
- **The weights.** The gaps enter the first layer through W_g: hidden x gap columns, zero at the start, no bias, drawn
  from no generator. So at the start each arm is zrckp bit for bit. A fit has 86 gap columns and 11,008 gap weights, as
  PSMK's and PSMH's fits had.
- **No message passing.** As in PSM and PSMH, the input aggregates over the question's rows, but only over the cache's
  fixed columns, as zret's z-scores do. No learned value and no row's score enters it, and it is computed before any row
  is scored. SEMB (whose columns the model computes) and the KT input take no gap. A block whose columns are not the
  batch's own is refused.

## 4. The rule (PSMK's section 3, with PSMS's names), for each arm on its own

- **The re-call.** Each of an arm's 12 reads (2 splits x 6 datasets) is called again under WQK's floors on zrckp
  (`outputs/wqk/floors-zrckp.json`, status OK, WQK's), by screen_recall.recall on the filed call and the R@5
  difference. A read's base R@5 must equal the floors record's seed-0 R@5 for that read. Otherwise the arm's verdict is
  INCOMPLETE.
- **The verdict**, by zpsm.py's own verdict code with the arm's PSMS names:
  - **ADOPT_FOR_FULL_RUN** when no read is LOSS and at least one is GAIN;
  - **NOT_ADOPTED** otherwise;
  - **INCOMPLETE** when a comparison or the floors are missing, are not WQK's or the arm's, or do not hold 12 reads.
- One arm's verdict does not change the other's.
- **Beside, deciding nothing:** the FC@5 and hit@1 differences per read, and each fit's train and read seconds.

## 5. What follows (fixed now, before any zrckpg1 or zrckph1 number)

Each next form gets its own declared file before its fit. Each may declare the T4 again, with WQK's bases and floors.
- **One arm ADOPT_FOR_FULL_RUN → its input on the GNN:** zspkg1 (zrckpg1's) or zspkh1 (zrckph1's), so that F1's
  controlled pair keeps one input set. Then the pair's full run under WQ.
- **Both ADOPT_FOR_FULL_RUN →** the arm with more GAIN reads goes on, as above. On a tie, the arm whose twelve R@5
  differences sum higher goes on.
- **Neither (both NOT_ADOPTED) → zrckpu1:** PSM's section 6 form for a GAIN and a LOSS, the row's lift above a uniform
  pool, log(N_q p_i) floored at 0, divided by 10. Undivided, it reaches log N_q (about 7.5 for a pool of 1,800 rows).
  Divided, it stays under 1, like the divided gaps. It is the last gap form PSM's file names.
- **Either INCOMPLETE:** the missing piece reruns, and nothing is decided until it has.

Each arm's verdict record names, as its next, the other arm's verdict and this section.

## 6. How it runs

**Two sessions, one per arm**, each a new version of one of KG's two kernels. That keeps KG's pinned image.
- zrckpg1 on mpr-kg: GPU 0 `scr-psms-zrckpg1-hp`, GPU 1 `scr-psms-zrckpg1`.
- zrckph1 on mpr-kg-g6b2: GPU 0 `scr-psms-zrckph1-hp`, GPU 1 `scr-psms-zrckph1`.

Where a session or a fit runs decides nothing. A session never queues both arms' fits. No fit starts after 8 hours.
- **Datasets:** KG's nine, WQK's three groups (mpr-wqk-1, -2, -3) and the code bundle. They are checked and linked by
  WQK's session code, unchanged.
- **Before anything runs:** WQK's verdict (`outputs/wqk/base-verdict.json`) must be BASE_READY under WQK's
  declaration, and the clone must be its commit.
- **Before any fit:** WQK's overlay (W9) and gate (W10), each in its own process. An overlay or gate that does not exit
  0 stops the session.
- **Each fit**, in its own process: `zpsms.py fit --name N --rows-cap 65536`. This is zpsmk's fit under the arm's PSMS
  names, with zpsm's arm swapped for the fit's arm. It refuses a name not PSMS's, a device that is not a T4 on KG's
  image, another rows cap, a gate that did not pass, and a WQK verdict that is not BASE_READY. Its timing record keeps
  zpsmk's file name, `psmk_timing.json`, with PSMS's declaration inside. Its messages keep zpsmk's wording, with PSMS's
  names in them.
- **Time and memory.** As PSMH's: about 37 minutes per session on two T4s, and about 4.6 GB of RAM per fit.

**On the laptop, after each session:**

    python outputs/mp_unified/zpsms.py take --from <the kernel's output folder>
    python outputs/mp_unified/zpsms.py compare-all --arm <zrckpg1 or zrckph1>
    python outputs/mp_unified/zpsms.py verdict --arm <zrckpg1 or zrckph1>

These are PSMK's take, compare-all and verdict (its section 4), with the arm's PSMS names. take finds the arm from the
session's fits, and refuses an output that holds fits of both arms. Outputs:
- `outputs/screen/scr-psms-<arm>.json` and `scr-psms-<arm>-hp.json`;
- `outputs/psms/<arm>-verdict.json` and `.md`;
- each session's gate and overlay under `outputs/psms/sessions/<the gate's sha256[:12]>/`.

**The stub** (`zpsms_session.py stub`) refuses what PSMK's refuses, and the two arms' fits together. It also refuses a
HEAD that does not track zpsms.py, zpsms_session.py, zpsmh.py and this file.

**What is committed:**
- Committed: this file, the code, the four compares and the two verdicts.
- Never committed: the fits' folders, weights, reads and logs, session.json, the datasets' manifests, and
  `outputs/psms/sessions`.

## 7. Self-tests, passed before this commit

`python outputs/mp_unified/zpsms.py --selftest` and `python outputs/mp_unified/zpsms_session.py --selftest` pass on
the laptop. They check:
1. **The names.**
   - Four fits on WQ's two splits, distinct from PSMH's, PSMK's, PSM's, WQK's and WQ's. The bases are WQK's seed-0
     zrckp fits, and the floors are WQK's. Each arm has its own verdict record.
   - Under an arm's PSMS names, zpsmk's names are the arm's, and zpsm's arm, input and next forms are the arm's. zpsmk's
     own check of zpsm's arm passes.
   - All are restored after, on a failure too. A set of fits from both arms, or none, is refused.
2. **The arms.**
   - zrckpg1 is zrckpg's class and zrckph1 is zrckph's, each with its gap divided by 10. zrckph's, zrckpg's, zrckp's,
     zrc's and zspk's arms are unchanged.
   - From the same seed, each arm's state is zrckpg's, and its forward is zrckp's bit for bit: in eval, and in training
     with the same dropout draws.
3. **The gaps.**
   - Each arm's gaps are its undivided arm's divided by 10, bit for bit, and within 1e-5 of the float64 reference
     divided by 10.
   - All lie in [-1, 0]. zrckph1's pool best is 0 in every column, and a dropped block is zero.
   - No learned value, SEMB input or KT column moves the gaps. One question's columns move only its own gaps, and a row
     permutation permutes them. A learned block is refused.
   - Two toy fits of each arm move W_g and are bit-identical.
4. **The guards, stamps, take, compare-all and verdict**, as PSMH's self-test checks them, for each arm:
   - the other arm's, PSMH's, PSMK's, PSM's and WQK's names are refused, and so is zrckpg under PSMS's prefix;
   - the stamps carry PSMS's block, not PSMK's or PSMH's, and the psm block names PSMS and the arm's divided gap;
   - take files the gate under `outputs/psms/sessions`, and refuses an output that holds both arms' fits;
   - the verdict's cases: for each arm, ADOPT_FOR_FULL_RUN (its GNN form), and NOT_ADOPTED with a GAIN and a LOSS and
     with no GAIN (the other arm's verdict, then zrckpu1). Also INCOMPLETE for eleven reads, a compare not the arm's
     (the other arm's and PSMH's among them), a compare made on another device, a fit and its base on two devices,
     floors not WQK's, a base R@5 off the floors, and a missing compare.
5. **The sessions**, on a toy repository with no zpsmk.py:
   - for each arm, the overlay and gate run first, then `zpsms.py fit --name N --rows-cap 65536`;
   - a failing overlay or gate stops it; the names, the two arms in one session, and the rows cap are checked;
   - the stub runs zpsms_session.py for either arm, and refuses a HEAD that tracks PSMH's files but not PSMS's.
6. **Nothing is written** under PSMS's, PSMH's or PSMK's real folders.

The text encoder stays frozen. Test splits are never read. There is no LLM, no topic-entity help and no new graph.
webqsp trains under WQ's ruling only.

## 8. Results

None yet.
