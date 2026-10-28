# PSMU: the lift above a uniform pool (zrckpu and zrckpu1) on Kaggle's T4

Declared 28 October 2026, after PSMS's verdicts (0510738) and before any zrckpu or zrckpu1 fit. PSMK's, PSMH's and
PSMS's compares show WQK's seed-0 zrckp R@5 on the twelve reads, and their own. zrckpu1's form was fixed before any
number by PSMS's section 5 ([PSMS_GAPS_AT_Z_SCALE_ON_T4.md](PSMS_GAPS_AT_Z_SCALE_ON_T4.md)); zrckpu, the same lift
undivided, is the form PSM's own section 6 names ([PSM_POOL_SOFTMAX_INPUT.md](PSM_POOL_SOFTMAX_INPUT.md)). Their device,
bases, floors and rule are PSMK's (sections 2 and 3 of [PSMK_PSM_ON_T4.md](PSMK_PSM_ON_T4.md)), which PSMS's section 5
lets a next form declare again. This file adds the undivided arm, one session per arm, and the next forms (section 5).
Code: `outputs/mp_unified/zpsmu.py` and `outputs/mp_unified/zpsmu_session.py`.

## 1. Why this, now

- **PSMS is NOT_ADOPTED in both arms** (0510738): zrckpg1 (the soft gap divided by 10) has 0 GAIN and 2 LOSS, and
  zrckph1 (the hard gap divided by 10) has 0 GAIN and 0 LOSS, under WQK's floors on zrckp. For neither adopted, PSMS's
  section 5 fixed the next form before any number: "zrckpu1: PSM's section 6 form for a GAIN and a LOSS, the row's lift
  above a uniform pool, log(N_q p_i) floored at 0, divided by 10."
- **Why zrckpu beside it.** PSMS found that dividing a gap by 10 removes most of its losses, but no form at either
  scale gained a read. The lift is a different input from the gaps, not only a different scale: it is 0 for every row
  at or below a uniform pool's share, and only rows above that share carry a value. Its undivided form is the one PSM's
  section 6 names. Screening it beside zrckpu1 asks of the lift what PSMS asked of the gaps, at both scales and in one
  round; one arm alone answers half of it.
- **A T4 each.** Both of KG's kernels are free after PSMS. G6RW needs one only once its files, typed links and encoder
  are in place. Each session costs about 34 minutes (PSMS's took 2,057 s and 1,969 s).

## 2. What changes, and what does not

| | PSMS | PSMU |
|---|---|---|
| arms | zrckpg1: PSM's soft gap divided by 10; zrckph1: PSMH's hard gap divided by 10 | zrckpu: the lift above a uniform pool; zrckpu1: the lift divided by 10 |
| fits | `scr-psms-zrckpg1`, `scr-psms-zrckpg1-hp`, `scr-psms-zrckph1`, `scr-psms-zrckph1-hp` | `scr-psmu-zrckpu`, `scr-psmu-zrckpu-hp`, `scr-psmu-zrckpu1`, `scr-psmu-zrckpu1-hp` |
| sessions | one per arm | one per arm |
| stamp key | `psms` | `psmu` |
| records | `outputs/psms/zrckpg1-verdict`, `outputs/psms/zrckph1-verdict` | `outputs/psmu/zrckpu-verdict`, `outputs/psmu/zrckpu1-verdict` |

Everything else is PSMK's, unchanged:
- the device: a T4 in a Kaggle kernel on KG's image;
- WQK's roots, with WQK's mixed KT input;
- the reads: the six s1eval carves at p@swa, with `--rows-cap 65536`;
- the bases: `scr-wqk-zrckp-s0` and `scr-wqk-zrckp-s0-hp`;
- the floors: `outputs/wqk/floors-zrckp.json`;
- the verdict code: zpsm's;
- the code: zpsmk.py and zpsmk_session.py, imported unchanged and run under each arm's PSMU names by zpsmu.py and
  zpsmu_session.py. zpsm.py, zpsmk.py, zpsmk_session.py, zpsmh.py, zpsmh_session.py, zpsms.py and zpsms_session.py do
  not change.

## 3. The arms

For each fixed block b (every block but SEMB), each of its columns c, and each row i of question q, with N_q the
question's rows:

    zrckpu:   u_ic = max(z_ic - log sum_{j in q} exp(z_jc) + log N_q, 0) * keep_qb
    zrckpu1:  u_ic = max(z_ic - log sum_{j in q} exp(z_jc) + log N_q, 0) / 10 * keep_qb

- **z** is zret's z-score of the column (`lean_screen3.seg_zscore_ref`), against the question's retrieved rows, as in
  PSM, PSMH and PSMS.
- **The lift.** u is log(N_q p_i), with p_i the softmax of z at temperature 1 over the question's N_q rows (the rows
  PSM's softmax runs over), floored at 0. A row at or below a uniform pool's share (p_i <= 1/N_q) is at 0. A row above
  it is at log N_q + log p_i, so at most log N_q (about 7.5 for a pool of 1,800 rows). The lift is computed from zpsm's
  unfloored log-softmax plus log N_q, then floored at 0. Wherever PSM's soft gap is above its floor of -10, the lift is
  that gap plus log N_q, floored at 0, bit for bit.
- **The division.** zrckpu1's lift is zrckpu's divided by 10, bit for bit. So it lies in [0, log(N_q) / 10], which is
  under 1 for any pool under e^10 (about 22,026) rows, like the divided gaps.
- **The weights.** The lifts enter the first layer through W_g: hidden x gap columns, zero at the start, no bias, drawn
  from no generator. So at the start each arm is zrckp bit for bit. A fit has 86 lift columns and 11,008 weights, as
  PSMK's, PSMH's and PSMS's fits had gap columns and gap weights.
- **No message passing.** As in PSM, PSMH and PSMS, the input aggregates over the question's rows, but only over the
  cache's fixed columns, as zret's z-scores do. No learned value and no row's score enters it, and it is computed
  before any row is scored. SEMB (whose columns the model computes) and the KT input take no lift. A block whose columns
  are not the batch's own is refused.

## 4. The rule (PSMK's section 3, with PSMU's names), for each arm on its own

- **The re-call.** Each of an arm's 12 reads (2 splits x 6 datasets) is called again under WQK's floors on zrckp
  (`outputs/wqk/floors-zrckp.json`, status OK, WQK's), by screen_recall.recall on the filed call and the R@5
  difference. A read's base R@5 must equal the floors record's seed-0 R@5 for that read. Otherwise the arm's verdict is
  INCOMPLETE.
- **The verdict**, by zpsm.py's own verdict code with the arm's PSMU names:
  - **ADOPT_FOR_FULL_RUN** when no read is LOSS and at least one is GAIN;
  - **NOT_ADOPTED** otherwise;
  - **INCOMPLETE** when a comparison or the floors are missing, are not WQK's or the arm's, or do not hold 12 reads.
- One arm's verdict does not change the other's.
- **Beside, deciding nothing:** the FC@5 and hit@1 differences per read, and each fit's train and read seconds.

## 5. What follows (fixed now, before any zrckpu or zrckpu1 number)

Each next form gets its own declared file before its fit. Each may declare the T4 again, with WQK's bases and floors.
- **One arm ADOPT_FOR_FULL_RUN → its input on the GNN:** zspku (zrckpu's) or zspku1 (zrckpu1's), so that F1's
  controlled pair keeps one input set. Then the pair's full run under WQ.
- **Both ADOPT_FOR_FULL_RUN →** the arm with more GAIN reads goes on, as above. On a tie, the arm whose twelve R@5
  differences sum higher goes on.
- **Neither (both NOT_ADOPTED) → the pool-softmax family closes for the MLP's input.** The soft gap, the hard gap and
  the lift, each undivided and divided by 10, will then all be NOT_ADOPTED under WQK's floors, so zrckp's input stays.
  The next forms under WQ go on, each in its own declared file before its fit:
  1. G6r under WQ on the T4 (WQK's section 9, item 1);
  2. U2n's zrckpt refit on U2s's pick under WQ (WQP's section 7);
  3. R2g and pool attention under WQ (WQ's section 9, through WQK);
  4. RRF5T's agreement as the pools' budget, and P1h's cut over the walk's mass (WQP's section 7).
- **Either INCOMPLETE:** the missing piece reruns, and nothing is decided until it has.

Each arm's verdict record names, as its next, the other arm's verdict and this section.

## 6. How it runs

**Two sessions, one per arm**, each a new version of one of KG's two kernels. That keeps KG's pinned image.
- zrckpu on mpr-kg: GPU 0 `scr-psmu-zrckpu-hp`, GPU 1 `scr-psmu-zrckpu`.
- zrckpu1 on mpr-kg-g6b2: GPU 0 `scr-psmu-zrckpu1-hp`, GPU 1 `scr-psmu-zrckpu1`.

Where a session or a fit runs decides nothing. A session never queues both arms' fits. No fit starts after 8 hours.
- **Datasets:** KG's nine, WQK's three groups (mpr-wqk-1, -2, -3) and the code bundle. They are checked and linked by
  WQK's session code, unchanged.
- **Before anything runs:** WQK's verdict (`outputs/wqk/base-verdict.json`) must be BASE_READY under WQK's
  declaration, and the clone must be its commit.
- **Before any fit:** WQK's overlay (W9) and gate (W10), each in its own process. An overlay or gate that does not exit
  0 stops the session.
- **Each fit**, in its own process: `zpsmu.py fit --name N --rows-cap 65536`. This is zpsmk's fit under the arm's PSMU
  names, with zpsm's arm swapped for the fit's arm. It refuses a name not PSMU's, a device that is not a T4 on KG's
  image, another rows cap, a gate that did not pass, and a WQK verdict that is not BASE_READY. Its timing record keeps
  zpsmk's file name, `psmk_timing.json`, with PSMU's declaration inside. Its messages keep zpsmk's wording, with PSMU's
  names in them.
- **Time and memory.** As PSMS's: about 34 minutes per session on two T4s, and about 4.6 GB of RAM per fit.

**On the laptop, after each session:**

    python outputs/mp_unified/zpsmu.py take --from <the kernel's output folder>
    python outputs/mp_unified/zpsmu.py compare-all --arm <zrckpu or zrckpu1>
    python outputs/mp_unified/zpsmu.py verdict --arm <zrckpu or zrckpu1>

These are PSMK's take, compare-all and verdict (its section 4), with the arm's PSMU names. take finds the arm from the
session's fits, and refuses an output that holds fits of both arms. Outputs:
- `outputs/screen/scr-psmu-<arm>.json` and `scr-psmu-<arm>-hp.json`;
- `outputs/psmu/<arm>-verdict.json` and `.md`;
- each session's gate and overlay under `outputs/psmu/sessions/<the gate's sha256[:12]>/`.

**The stub** (`zpsmu_session.py stub`) refuses what PSMK's refuses, and the two arms' fits together. It also refuses a
HEAD that does not track zpsmu.py, zpsmu_session.py, zpsmh.py and this file.

**What is committed:**
- Committed: this file, the code, the four compares and the two verdicts.
- Never committed: the fits' folders, weights, reads and logs, session.json, the datasets' manifests, and
  `outputs/psmu/sessions`.

## 7. Self-tests, passed before this commit

`python outputs/mp_unified/zpsmu.py --selftest` and `python outputs/mp_unified/zpsmu_session.py --selftest` pass on
the laptop. They check:
1. **The names.**
   - Four fits on WQ's two splits, distinct from PSMS's, PSMH's, PSMK's, PSM's, WQK's and WQ's. The bases are WQK's
     seed-0 zrckp fits, and the floors are WQK's. Each arm has its own verdict record, and PSMU's stage, key, sessions
     folder and verdicts are none of PSMS's, PSMH's or PSMK's.
   - Under an arm's PSMU names, zpsmk's names are the arm's, and zpsm's arm, input and next forms are the arm's. zpsmk's
     own check of zpsm's arm passes.
   - All are restored after, on a failure too. A set of fits from both arms, or none, or a PSMS fit, is refused.
2. **The arms.**
   - zrckpu is zrckpg's class with the lift, and zrckpu1 is zrckpu's with the lift divided by 10. zrckph's, zrckpg's,
     zrckp's, zrc's and zspk's arms are unchanged.
   - From the same seed, each arm's state is zrckpg's, and its forward is zrckp's bit for bit: in eval, and in training
     with the same dropout draws.
3. **The lifts.**
   - zrckpu's are within 1e-4 of a float64 reference, and zrckpu1's are zrckpu's divided by 10, bit for bit.
   - All lie in [0, log N_q]. Wherever PSM's soft gap is above -10, the lift is that gap plus log N_q, floored at 0, bit
     for bit. A constant column's lift is 0 (below 1e-5), a dropped block's is zero, and every question's live column
     that varies has a positive lift somewhere. A three-row check gives 0, log 2 and 0.
   - No learned value, SEMB input or KT column moves the lifts. One question's columns move only its own lifts, and a
     row permutation permutes them. A learned block is refused.
   - Two toy fits of each arm move W_g and are bit-identical.
4. **The guards, stamps, take, compare-all and verdict**, as PSMS's self-test checks them, for each arm:
   - the other arm's, PSMS's, PSMH's, PSMK's, PSM's and WQK's names are refused, and so is zrckpg under PSMU's prefix;
   - the stamps carry PSMU's block, not PSMK's, PSMH's or PSMS's, and the psm block names PSMU and the arm's lift;
   - take files the gate under `outputs/psmu/sessions`, and refuses an output that holds both arms' fits, a PSMS fit
     or a PSMK fit;
   - the verdict's cases: for each arm, ADOPT_FOR_FULL_RUN (its GNN form), and NOT_ADOPTED with a GAIN and a LOSS and
     with no GAIN (the other arm's verdict, then the family's close). Also INCOMPLETE for eleven reads, a compare not
     the arm's (the other arm's and PSMS's among them), a compare made on another device, a fit and its base on two
     devices, floors not WQK's, a base R@5 off the floors, and a missing compare.
5. **The sessions**, on a toy repository with no zpsmk.py:
   - for each arm, the overlay and gate run first, then `zpsmu.py fit --name N --rows-cap 65536`;
   - a failing overlay or gate stops it; the names, the two arms in one session, and the rows cap are checked;
   - the stub runs zpsmu_session.py for either arm, and refuses a HEAD that tracks PSMS's files but not PSMU's.
6. **Nothing is written** under PSMU's, PSMS's, PSMH's or PSMK's real folders.

The text encoder stays frozen. Test splits are never read. There is no LLM, no topic-entity help and no new graph.
webqsp trains under WQ's ruling only.

## 8. Results

**The sessions.** Two sessions at 7009588 on 28 October (UTC), one arm each, both launched at about 03:03; both had
ended by 03:44:
- zrckpu ran on version 8 of mpr-kg. WQK's overlay and gate exited 0 (144 s and 81 s). Both fits exited 0 on Tesla
  T4s: `scr-psmu-zrckpu-hp` on GPU 0 in 1,783 s, and `scr-psmu-zrckpu` on GPU 1 in 1,756 s. The session took 2,115 s.
  Its gate and overlay are filed under `outputs/psmu/sessions/a291b2033245` (not committed).
- zrckpu1 ran on version 5 of mpr-kg-g6b2. WQK's overlay and gate exited 0 (120 s and 82 s). Both fits exited 0 on
  Tesla T4s: `scr-psmu-zrckpu1-hp` on GPU 0 in 1,837 s, and `scr-psmu-zrckpu1` on GPU 1 in 1,840 s. The session took
  2,127 s. Its gate and overlay are filed under `outputs/psmu/sessions/ce46ca364f54` (not committed).

| fit | train s | read s | peak RAM GB | lift columns | lift weights |
|---|---:|---:|---:|---:|---:|
| scr-psmu-zrckpu | 1,258.4 | 484.5 | 4.48 | 86 | 11,008 |
| scr-psmu-zrckpu-hp | 1,294.6 | 477.2 | 4.58 | 86 | 11,008 |
| scr-psmu-zrckpu1 | 1,341.0 | 487.5 | 4.49 | 86 | 11,008 |
| scr-psmu-zrckpu1-hp | 1,338.3 | 485.6 | 4.61 | 86 | 11,008 |

**zrckpu's verdict: NOT_ADOPTED** (`outputs/psmu/zrckpu-verdict.json` and `.md`), with 0 GAIN and 1 LOSS under WQK's
floors on zrckp.
- **LOSS:** WQ-hotpotqa webqsp, -0.0101 against a floor of 0.0075.
- **Filed calls, called WITHIN.** lean_screen filed three other reads as GAIN or LOSS; under WQK's floors all three are
  WITHIN:
  - WQ-musique metaqa: filed LOSS, -0.0221, against a floor of 0.0621;
  - WQ-musique musique: filed LOSS, -0.0143, against a floor of 0.0566;
  - WQ-hotpotqa metaqa: filed GAIN, +0.0111, against a floor of 0.0489.

**zrckpu1's verdict: NOT_ADOPTED** (`outputs/psmu/zrckpu1-verdict.json` and `.md`), with 0 GAIN and 0 LOSS under
WQK's floors on zrckp.
- **Filed calls, called WITHIN.** lean_screen filed one read as a GAIN; under WQK's floors it is WITHIN:
  - WQ-hotpotqa metaqa: filed GAIN, +0.0415 (interval +0.0363 to +0.0467), against a floor of 0.0489.

**Beside: the six forms.** These are R@5 differences from WQK's seed-0 zrckp fit of each split, read by read. A call is
shown only where it is not WITHIN under WQK's floors.

| split | dataset | read | soft (PSMK) | soft / 10 (zrckpg1) | hard (PSMH) | hard / 10 (zrckph1) | lift (zrckpu) | lift / 10 (zrckpu1) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| WQ-musique | metaqa | zero-shot | +0.0293 | +0.0224 | -0.0380 | -0.0144 | -0.0221 | -0.0032 |
| WQ-musique | squad | in-domain | -0.0027 | -0.0015 | -0.0012 | +0.0008 | -0.0005 | -0.0008 |
| WQ-musique | musique | zero-shot | +0.0076 | -0.0359 | +0.0077 | +0.0025 | -0.0143 | -0.0036 |
| WQ-musique | hotpotqa | in-domain | -0.0254 **LOSS** | -0.0044 | -0.0063 | -0.0030 | -0.0022 | -0.0019 |
| WQ-musique | 2wiki | in-domain | -0.0115 **LOSS** | -0.0028 | -0.0072 | -0.0009 | -0.0028 | -0.0001 |
| WQ-musique | webqsp | in-domain | -0.0133 **LOSS** | -0.0080 | -0.0182 **LOSS** | -0.0023 | -0.0042 | -0.0046 |
| WQ-hotpotqa | metaqa | zero-shot | +0.0026 | +0.0183 | +0.0355 | +0.0177 | +0.0111 | +0.0415 |
| WQ-hotpotqa | squad | in-domain | -0.0049 | -0.0003 | +0.0011 | -0.0006 | +0.0014 | +0.0010 |
| WQ-hotpotqa | musique | in-domain | -0.0175 **LOSS** | -0.0008 | +0.0017 | -0.0010 | -0.0001 | -0.0012 |
| WQ-hotpotqa | hotpotqa | zero-shot | -0.0145 **LOSS** | -0.0116 **LOSS** | -0.0056 | -0.0005 | +0.0035 | +0.0043 |
| WQ-hotpotqa | 2wiki | in-domain | -0.0201 **LOSS** | -0.0018 | -0.0036 | +0.0004 | -0.0021 | -0.0003 |
| WQ-hotpotqa | webqsp | in-domain | -0.0142 **LOSS** | -0.0105 **LOSS** | -0.0139 **LOSS** | -0.0053 | -0.0101 **LOSS** | -0.0067 |
| | | sum | -0.0846 | -0.0369 | -0.0480 | -0.0066 | -0.0424 | +0.0244 |
| | | sum without metaqa | -0.1165 | -0.0776 | -0.0455 | -0.0099 | -0.0314 | -0.0139 |

Dividing the lift by 10 removes its one loss, as dividing each gap removed most of its losses. The lift / 10 is the
only form whose twelve differences sum above 0 (+0.0244), but that sum rests on one read: WQ-hotpotqa's zero-shot
metaqa, +0.0415, which is under its floor of 0.0489. Without the two metaqa reads, every form's other ten differences
sum below 0 (-0.0139 for the lift / 10). WQ-hotpotqa's metaqa read rose under all six forms (+0.0026 to +0.0415), each
time under its floor; WQ-musique's metaqa read rose under three forms and fell under three. Across the six forms no read
is a GAIN under WQK's floors, and 12 are LOSS (7, 2, 2, 0, 1 and 0).

**The family closes for the MLP's input.** Neither arm is adopted, so section 5 applies. The soft gap, the hard gap and
the lift, each undivided and divided by 10, are all NOT_ADOPTED under WQK's floors, and zrckp's input stays. This closes
these six forms. Pool attention is a different form and is still item 3 below.

**Next.** Per section 5, the next forms under WQ go on, each in its own declared file before its fit:
1. G6r under WQ on the T4 (WQK's section 9, item 1);
2. U2n's zrckpt refit on U2s's pick under WQ (WQP's section 7);
3. R2g and pool attention under WQ (WQ's section 9, through WQK);
4. RRF5T's agreement as the pools' budget, and P1h's cut over the walk's mass (WQP's section 7).
