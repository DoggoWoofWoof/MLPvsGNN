# PSMK: PSM's screen on Kaggle's T4

Declared 28 October 2026, before any PSM or PSMK fit ran and before any WQK number. PSM
([PSM_POOL_SOFTMAX_INPUT.md](PSM_POOL_SOFTMAX_INPUT.md), 6ce6c8b) declared its two seed-0 fits on the host's GPU,
called under WQ's floors. This file moves that screen to Kaggle's T4, under WQK's base set
([WQK_WQ_ON_T4.md](WQK_WQ_ON_T4.md)). It changes PSM's device, bases and floors, and nothing else. The arm, its input,
the rule and the next forms stay PSM's. Code: `outputs/mp_unified/zpsmk.py` and `outputs/mp_unified/zpsmk_session.py`.

## 1. Why this, now

- **The host is offline**, since 22:17 on 27 October. WQ's host base set has not started, and PSM's fits queue behind
  its summary item. The user, 28 October: "the host might go on and off so keep the queue full proper utilization and
  keep stuff ready".
- **WQK provides for this.** WQK's NEXT reads: "a later WQ stage may declare the T4 as its device: WQK's seed-0 fits are
  then its bases and WQK's floors its floors". PSMK is that declaration for PSM.
- **What it decides.** PSM's screen verdict, on the T4. PSMK gives no second verdict to the same question: when PSMK's
  session is launched, PSM's host items (`psm-*`) are withdrawn from the host's queue. Until then they stay queued, as
  PSM declared.

## 2. What changes, and what does not

| | PSM (host) | PSMK (T4) |
|---|---|---|
| arm | zrckpg (zpsm.py) | the same, imported unchanged |
| fits | `scr-psm-zrckpg`, `scr-psm-zrckpg-hp` | `scr-psmk-zrckpg`, `scr-psmk-zrckpg-hp` |
| device | the host's GPU | a T4 in a Kaggle kernel on KG's image (WQK, section 3) |
| roots | WQ's | WQK's (zwqk.on_wqk, with WQK's mixed KT input) |
| when it trains | after WQ's summary is BASE_READY | after WQK's verdict is BASE_READY, committed before the stub |
| reads | the six s1eval carves at p@swa | the same, with `--rows-cap 65536` (whole questions, as WQK's) |
| bases | `scr-wq-zrckp-s0[-hp]` | `scr-wqk-zrckp-s0[-hp]` |
| floors | `outputs/wq/floors-zrckp.json` | `outputs/wqk/floors-zrckp.json` |
| record | `outputs/psm/screen-verdict` | `outputs/psmk/screen-verdict` |

A fit and its base are read on one device, and the compare checks it: both fits' stamps must name the same T4 and
image as the base's.

## 3. The rule (PSM's section 5, with WQK's floors)

- **The re-call.** Each of the 12 reads (2 splits x 6 datasets) is called again under WQK's floors on zrckp
  (`outputs/wqk/floors-zrckp.json`, status OK, WQK's), by screen_recall.recall on the filed call and the R@5
  difference. A read's base R@5 must equal the floors record's seed-0 R@5 for that read. Otherwise the verdict is
  INCOMPLETE.
- **The verdict**, by zpsm.py's own verdict code with PSMK's names:
  - **ADOPT_FOR_FULL_RUN** when no read is LOSS and at least one is GAIN;
  - **NOT_ADOPTED** otherwise;
  - **INCOMPLETE** when a comparison or the floors are missing, are not WQK's or PSMK's, or do not hold 12 reads.
- **Beside, deciding nothing:** the FC@5 and hit@1 differences per read, and each fit's train and read seconds.
- **What follows** is PSM's section 6, unchanged. Each next form gets its own declared file. It may declare the T4
  again, with WQK's bases and floors.

## 4. How it runs

**One session**, a new version of whichever of KG's two kernels is free (mpr-kg or mpr-kg-g6b2). That keeps KG's
pinned image.
- **Datasets:** KG's nine, WQK's three groups (mpr-wqk-1, -2, -3) and the code bundle. The datasets are WQK's, checked
  and linked by WQK's session code (zwqk_session's find, plan, verify and link, unchanged).
- **Before anything runs:** the session checks that WQK's verdict (`outputs/wqk/base-verdict.json`) is BASE_READY
  under WQK's declaration, and that the clone is its commit.
- **Before any fit:** WQK's overlay (W9) and gate (W10), each in its own process (`zwqk.py overlay`, `zwqk.py gate`).
  An overlay or gate that does not exit 0 stops the session.
- **The queues:**
  - GPU 0: `scr-psmk-zrckpg-hp`;
  - GPU 1: `scr-psmk-zrckpg`.

  No fit starts after 8 hours. Where a fit runs decides nothing.
- **Each fit**, in its own process: `zpsmk.py fit --name N --rows-cap 65536`. It refuses a name not PSMK's, a device
  that is not a T4 on KG's image, a gate that did not pass, and a WQK verdict that is not BASE_READY. It trains zrckpg
  with zp1c3p's train under zpsm's arm swap, stamps the fit, then reads it on the device it was fitted on. A read is
  stamped only when it exits 0. A read PSMK stamped is kept; a fit trained and stamped but not read is only read.
- **Time.** About two of WQK's zrckp fits, so about an hour on two T4s. It costs nothing (the week's Kaggle GPU hours).

**On the laptop, after the session:**

    python outputs/mp_unified/zpsmk.py take --from <the kernel's output folder>
    python outputs/mp_unified/zpsmk.py compare-all
    python outputs/mp_unified/zpsmk.py verdict

- **take** copies each PSMK fit's records and reads into `outputs/screen/fits/<name>`, never models.pt. It copies the
  session's gate.json and overlay.json to `outputs/psmk/sessions/<the gate's sha256[:12]>/`. A fit without PSMK's
  stamp and a read on the device it was fitted on is left where it is and named. A folder that is not one of PSMK's
  fits, a file already there with other bytes, or an output without the session's gate and overlay stops the take.
- **compare-all** runs zpsm's compare of each fit against its split's WQK seed-0 zrckp fit alone. It requires both on
  one device and stamps the compare with PSMK's record and that device. A compare PSMK stamped before is kept.
  Outputs: `outputs/screen/scr-psmk-zrckpg[-hp].json`.
- **verdict** writes `outputs/psmk/screen-verdict.json` and `.md`. It exits 0 on ADOPT_FOR_FULL_RUN or NOT_ADOPTED
  and 2 after writing an INCOMPLETE record.

**The stub** (`zpsmk_session.py stub`) refuses a name or rows cap not PSMK's, a WQK verdict that is not BASE_READY, a
working tree that differs from HEAD, and a HEAD that is not main, that origin does not hold, or that does not track
PSMK's and WQK's code, WQK's ship.json, its records, its verdict and floors on zrckp, and this file.

**What is committed:**
- Committed: this file, the code, the two compares and the verdict.
- Never committed: the fits' folders, weights, reads and logs, session.json, the datasets' manifests, and
  `outputs/psmk/sessions`.

## 5. Self-tests, passed before this commit

`python outputs/mp_unified/zpsmk.py --selftest` and `python outputs/mp_unified/zpsmk_session.py --selftest` pass on
the laptop. They check:
1. **The names.** PSMK's fits are zpsm's arm on WQ's two splits, distinct from PSM's and WQK's. Each base is WQK's
   seed-0 zrckp fit of its split, and the floors path is WQK's. Under PSMK's names, zpsm's and zwq's names are PSMK's
   and WQK's; they are restored after, on a failure too.
2. **The guards.** A fit refuses:
   - another name;
   - a device that is not a T4 on KG's image;
   - another rows cap;
   - a WQK verdict that is missing, not BASE_READY, or not under WQK's declaration;
   - a gate that is missing or did not pass.

   A read also refuses a device other than its fit's.
3. **The stamps.** A trained fit's record carries:
   - the wq block at WQK's roots and declaration;
   - PSM's psm block naming this file: the gap columns and weights, and WQK's verdict;
   - PSMK's block: device, gate, overlay, check, the KT and chain records, and the code.

   zpsm's own record check passes on it under PSMK's names only.
4. **Take**, on fake fits. It:
   - copies records and reads but never models.pt, and files the session's gate and overlay by the gate's sha256;
   - leaves an unstamped fit, and one read on another device;
   - refuses a folder that is not PSMK's, a file already there with other bytes, and an output without the session's
     gate.
5. **compare-all**, with lean_screen's compare faked. It stamps each compare as PSMK's, with its device, and keeps it on
   a second run. It refuses a fit and its base on two devices.
6. **The verdict**, on fake compares and floors. It checks each outcome:
   - ADOPT_FOR_FULL_RUN, with a GAIN and no LOSS;
   - NOT_ADOPTED, with a LOSS, and with no GAIN;
   - INCOMPLETE, with any of: a missing compare; floors that are not WQK's; a base R@5 off the floors' seed-0 value;
     eleven reads; a compare not stamped as PSMK's, or not of PSM's form; a compare made on another device; a fit and
     its base on two devices.

   The record and its .md name PSMK, and WQK's bases and floors. zpsm's names are restored after.
7. **The session**, on a toy repository and toy datasets:
   - the names are refused unless each is PSMK's, once, and the rows cap unless it is 65536;
   - a WQK verdict that is missing or not BASE_READY, or a clone that differs from its commit, stops it before
     anything runs;
   - the overlay and gate run first, and a failing one stops the session before any fit;
   - each fit's command is `zpsmk.py fit --name N --rows-cap 65536`.

   The stub refuses: a WQK verdict that is not BASE_READY; an untracked declaration; a dirty tree; a HEAD that is not
   main; a HEAD that origin does not hold. With all of them held, it writes the verified bundle and the stub at HEAD.

The text encoder stays frozen. Test splits are never read. There is no LLM, no topic-entity help and no new graph.
webqsp trains under WQ's ruling only.

## 6. Results

**The session.**
- Version 5 of mpr-kg, at 9059de7, from 23:46 on 27 October to 00:21 on 28 October (UTC).
- WQK's overlay and gate exited 0 (106 s and 83 s).
- Both fits exited 0 on Tesla T4s: `scr-psmk-zrckpg-hp` on GPU 0 in 1,761 s, and `scr-psmk-zrckpg` on GPU 1 in
  1,819 s. The session took 2,087 s.
- Its gate and overlay are filed under `outputs/psmk/sessions/baed26a5bf32` (not committed).

| fit | train s | read s | peak RAM GB | gap columns | gap weights |
|---|---:|---:|---:|---:|---:|
| scr-psmk-zrckpg | 1,329.1 | 477.4 | 4.49 | 86 | 11,008 |
| scr-psmk-zrckpg-hp | 1,276.6 | 468.8 | 4.61 | 86 | 11,008 |

**The verdict: NOT_ADOPTED** (`outputs/psmk/screen-verdict.json` and `.md`), with 0 GAIN and 7 LOSS under WQK's floors
on zrckp.
- **LOSS:**
  - WQ-musique: hotpotqa (-0.0254), 2wiki (-0.0115), webqsp (-0.0133);
  - WQ-hotpotqa: musique (-0.0175), hotpotqa (-0.0145), 2wiki (-0.0201), webqsp (-0.0142).
- **Filed GAIN, called WITHIN.** lean_screen filed two reads as GAIN; under WQK's floors both are WITHIN:
  - WQ-musique metaqa: +0.0293, against a floor of 0.0621;
  - WQ-musique musique: +0.0076, against a floor of 0.0566.

The soft gap lowers R@5 on most in-domain reads in both splits. FC@5 falls with it on the seven LOSS reads, by 0.015
to 0.046.

**Next.** PSM's section 6 for NOT_ADOPTED with no GAIN is **zrckph**: the hard gap, z_i minus the pool's largest z,
floored at -10. It gets its own declared file before its fit, and may declare the T4 again with WQK's bases and floors.
