# R2t: R2's bridge heads on Kaggle's T4

Declared 27 October 2026. Code: `outputs/mp_unified/zr2t.py`. Its fits run in KG's sessions (`zkg_session.py`) on KG's
device. No R2 fit has run, on any device. Nothing in this file reads an R2 number.

## 1. Why this, now

- **R2 is the next GPU stage in the program's order.** The program of 24 October orders the GPU stages N2, G6, R2
  (zg6m and zrckpb), then A5cp, A7p and P5c5, then U2n's forms (docs/PROGRAM_2026_10_24.md, section 4.3). It says:
  "Kaggle's T4 runs once the user's token is set; Modal's credit is the rest", and "Bases and nulls are refit on each
  new device."
- **R2 has not run.** It was declared on 20 October for the lab host's card (docs/R2_BRIDGE_EM.md). The host went
  offline that day and has not come back. GL part 2 moved R2 to workspace 3's L40S (docs/GL_GPU_LINE_ON_L40S.md,
  section 3), which renews on 1 November.
- **KG made the T4 a device a stage can declare.** KG filed BASE_READY on 27 October (docs/KG_KAGGLE_T4_LINE.md,
  section 10): U2d's base build's zrckp and zspk with seeds 0, 1 and 2, and zg6 with seed 0, trained and read on one T4
  from one image, with the null compares and zg6 against zspk. KG's section 6: "After BASE_READY, a later stage may
  declare the T4 as its device. KG's fits are then its bases and KG's nulls its floors."
  - R2's two bases are there: zrckp's and zg6's seed-0 fits.
  - So are the two nulls R2 names: zrckp's, and zspk's moved onto zg6.
- **So R2 is decided on the T4 now**, by R2's own rule, constants, model and next forms, on the files KG filed. It does
  not wait for 1 November. GL part 2's R2 fits, if they run, are a second device's reading of R2 and decide nothing
  (section 7).
- **The risks, stated before any number.**
  - R2's arms have never trained on real data. zr2br.py's self-test ran them on a toy carve (R2, section 6), and
    zr2t.py's runs them only on fakes of rmatch's train and read. A fit that fails is named in session.json with its
    exit code and log, and its arm's verdict is INCOMPLETE.
  - R2's two limits stand unchanged (R2, section 4). KG's nulls measure zrckp's and zspk's seed spread, not zrckpb's or
    zg6m's. On the KB datasets the first stage's top 5 rarely holds a gold, so the bridges are seldom golds.

## 2. The arms, R2's unchanged

- **zr2br.py's two arms**, with no change to the model or its training (R2, sections 2 and 3):
  - **zrckpb**: zrckp, the lean MLP (zrc plus P1c3p's pointwise head on the eight KT columns), plus R2's head;
  - **zg6m**: zg6, G6's free GNN, plus R2's head on its final score.
- **R2's constants.** K = 5 bridges by the first stage's RRF; r = 32; LAM = 1; the prior 1/(60 + k); R2's own
  generator at seed + 5502; the head Linear(3, 16), GELU, Linear(16, 1), its last layer at zero; W_1's scale 0.1.
  59,537 parameters per arm. From the same seed, each arm starts as its base, bit for bit.
- **R2's EM term in training**, through zr2br.py's own wrapper: lean_gpu's listwise loss plus LAM times the batch mean
  of L_br. As in zr2br.py, a fit where no training batch carried the term stops unstamped.
- **Checks.** zr2t.py checks R2's constants before a fit trains and before it is read. Each fit's stamp holds the
  constants and the EM term for each epoch.
- **zrckpb's name.** Whether reading the five bridges' vectors keeps zrckpb inside the MLP track is still the user's
  ruling (R2, section 2). Until then every record calls it "zrckp + R2's bridge head", and no claim sets it against the
  GNN as the MLP. Its rule and verdict do not change with the ruling.
- **zg6m** is decided against zg6, whatever G6's verdict (R2, section 3). G6 is NOT_ADOPTED on the L40S (GL, section
  7), and so is KG's replication of G6's call on the T4 (KG, section 10). zg6 stays zg6m's base, as R2 fixed.

## 3. The fits and the device

Four fits on U2d's base build, seed 0:

| R2t fit | split | arm | its KG bases, all on the same T4 |
| --- | --- | --- | --- |
| `scr-r2t-zrckpb` | L-musique | zrckpb | `scr-kg-base-zrckp` (decides) |
| `scr-r2t-zrckpb-hp` | L-hotpotqa | zrckpb | `scr-kg-base-zrckp-hp` (decides) |
| `scr-r2t-zg6m` | L-musique | zg6m | `scr-kg-base-zg6` (decides); `scr-kg-base-zspk` (beside) |
| `scr-r2t-zg6m-hp` | L-hotpotqa | zg6m | `scr-kg-base-zg6-hp` (decides); `scr-kg-base-zspk-hp` (beside) |

- **The same path as KG's.** zr2t.py trains and reads through zu2d.py's train and read bodies, as zkg.py runs them:
  - U2d's base build's roots and gate: P1c2's construction with P1c3's KT input, whose KT arrays equal P1c3's on all
    eleven carves (U2d, section 11);
  - U2d's card (P1c2's card settings) and seed 0;
  - reads of p@swa on the six s1eval carves.

  Each fit's stamp records the device, the build, the pools, the KT, rmatch, zrc and zlink records, the base's head
  settings, R2's settings and constants, the EM term and the fit's KG bases.
- **Reads at 65,536 rows a batch, for both arms.** KG read every one of its fits at `--rows-cap 65536`. R2 set 65,536
  for zg6m's reads, and zrckpb's reads in R2 and GL took lean_gpu's default of 262,144. A batch holds whole questions,
  so the scores do not change (U2d, section 5). At 65,536, zrckpb is read as KG's zrckp was. zr2t.py refuses any other
  cap.
- **The device is KG's.** Train and read run only in a Kaggle kernel on a T4 from KG's image
  (kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3), checked before any data is read. A read must run on its fit's
  device, with R2's constants as declared.
- **Before R2t is called:**
  - KG's base verdict (outputs/kg/base-verdict.json) must be BASE_READY;
  - each arm's two fits must be on KG's device.
- **Memory.** KG's peaks on the T4 were 3.7 to 5.6 GB for zrckp and zspk, 4.7 GB training and 5.2 GB reading for zg6,
  and 10.3 GB for zg6-hp; the T4 has 15 GB. R2's head adds, per row, one 257 × 32 map and five products of width 32 (R2,
  section 2), and in training the logits the EM term reads. Each fit's r2t_timing.json records its peaks.

## 4. The compares, the nulls and the rule

**The rule is R2's** (R2, section 4). So are its constants, model and next forms (its sections 2 and 5). The files
change, as they did for GL:

| R2 as declared (the host's card) | R2t (Kaggle's T4, U2d's base build) |
| --- | --- |
| zrckpb's fits `scr-r2-zrckpb`, `-hp` | `scr-r2t-zrckpb`, `-hp` |
| zg6m's fits `scr-r2-zg6m`, `-hp` | `scr-r2t-zg6m`, `-hp` |
| zrckpb's base `scr-p1c3p-zrckp`, `-hp` | KG's `scr-kg-base-zrckp`, `-hp` |
| zg6m's base `scr-g6-zg6`, `-hp` | KG's `scr-kg-base-zg6`, `-hp` |
| zrckpb's null: N2's zrckp null | KG's zrckp null (`scr-kg-null-zrckp-s1`, `-s2`, `-s1-hp`, `-s2-hp`) |
| zg6m's null: N2's zspk null, moved onto zg6 through G6's compares | KG's zspk null (`scr-kg-null-zspk-*`), moved onto zg6 through KG's `scr-kg-zg6-vs-zspk`, `-hp` |
| each fit against its base's card fit, with step 1's fit beside | each fit against its KG base; zg6m's also against KG's zspk, beside |
| INCOMPLETE when N2's floors record is missing or not OK | INCOMPLETE when KG's base verdict is not BASE_READY, or KG's null or move compares fail zkg.py's checks |

- **KG's nulls are N2's design on the T4.** Each base arm was trained again with seeds 1 and 2, and each fit compared
  with its seed-0 fit (KG, section 5). A read's floor is screen_recall's max(0.0075, 2·sqrt((D1² + D2²)/2)), from the
  seed-1 and seed-2 R@5 differences; zkg.py's `kg_null` computes them.
- **The move onto zg6.** Each read's seed-0 R@5 becomes zg6's, through KG's compare of zg6 with zspk of the split. That
  compare's zspk column must equal the null's seed 0. D1, D2 and the floors stay zspk's. It is the stand-in R2 itself
  declared, and the one GL takes on the L40S.
- **Four compares:** `scr-r2t-zrckpb-vs-zrckp`, `scr-r2t-zg6m-vs-zg6` and their `-hp` twins. Each is lean_screen's
  compare (p@swa on s1eval), stamped with R2t's record. The fit and its bases must be on one device, so step 1's fit,
  from another device, is not set beside.
- **The calls, per arm.**
  1. screen_pair over the arm's two compares against its base: `outputs/r2t/r2t-<arm>-pair`.
  2. A re-call under its null: `outputs/r2t/r2t-<arm>-pair-recall`.

  Each compare's base column must be that base's read as KG filed it: zrckp's seed-0 R@5 as KG's zrckp null compares
  hold it, and zg6's as KG's zg6-vs-zspk compares hold it. screen_recall checks this exactly.
- **The rule, per arm.**
  - **ADOPT_FOR_FULL_RUN:** its re-call under its null has no LOSS on any of its twelve reads, and at least one GAIN.
    The twelve are six datasets times two fits; webqsp is read zero-shot in both, and so is each fit's held-out dataset.
  - **NOT_ADOPTED:** otherwise.
  - **INCOMPLETE:** any of these:
    - KG's base verdict is not BASE_READY;
    - a fit or compare of the arm is missing, or is not R2t's;
    - the arm's fits are not on KG's device;
    - a compare's base column is not that base's read as KG filed it;
    - KG's null or move compares fail zkg.py's checks;
    - one of the arm's re-calls, zg6m's beside one included, does not hold the twelve reads.

    The verdict names what is missing. One arm's INCOMPLETE does not hold back the other arm's call.

## 5. Beside, deciding nothing

- **zg6m against KG's zspk**, its compares' second base, through the same pair and a re-call under KG's zspk null
  without the move (`outputs/r2t/r2t-zg6m-vs-zspk-pair` and `-recall`). This sets R2's head on the free GNN against the
  lean GNN the program carries.
- **Each arm minus its base per read**: R@5 with its 95% question-bootstrap interval, FC@5 and hit@1. zg6m minus zspk
  is shown the same way.
- **The webqsp and metaqa reads** of each arm, and musique in-domain (L-hotpotqa) for both (R2, section 4).
- **R@5 per read on the T4** for rrf, zspk, zg6, zg6m, zrckp and zrckpb, with two MLP shares where the GNN's gain over
  rrf exceeds 0.002: (zrckp − rrf) / (zg6 − rrf), and (zrckpb − rrf) / (zg6m − rrf). Until the user rules on zrckpb,
  the second share is not an MLP-track number.
- **R2's EM term per epoch**, from each fit's stamp: L_br's mean, and the share of training questions with a term.
- **Seconds and memory** for each fit, beside KG's base of the same split.

Not set beside:
- the 7 October re-calls and N2's null, whose fits are the host's, another device;
- step 1's fit, also another device's.

## 6. What each outcome leads to (fixed now)

| an arm's re-call under its KG null | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on U2s's carried pools, then its full run on all six splits beside its base's, in its own declared file |
| NOT_ADOPTED, with at least one GAIN and a LOSS | R2g (R2, section 5): a learned per-question back-off, R2's head times 2 · sigmoid(a · c_q + b), the gate starting at 1 (one declared file) |
| NOT_ADOPTED, with no GAIN | R2e (R2, section 5): the edge form, bridge scores only for bridges linked to the row by zlink's edges, as fixed per-row link flags for the MLP and inside zg6's messages for the GNN (one declared file) |
| INCOMPLETE | what is missing is supplied (KG's base set, a fit, a compare or a re-call); nothing else changes |

- **When only one arm passes,** the other goes to its own next form. Neither is closed.
- **A zg6m refit** puts R2's head on zg6, or on G6's next form if G6 is not adopted. That refit's own file fixes which
  (R2, section 5).
- **Why U2s's pools.** R2 named P1g-b's pools for the refit. P1g-b was not adopted, and U2s (the walk-mass budget, one
  construction for all six datasets) is the pools stage the program now carries, as G6b2 and U2n name it.

No verdict here closes the idea (docs/SCREENS.md, section 3b). No model change is adopted on today's pools (the user's
rule of 13 October).

## 7. How it runs

- **In KG's sessions.** zkg_session.py runs each queued fit through its stage's script, each in its own process:
  `zkg.py fit` for scr-kg- names, `zg6b.py fit` for scr-g6b- names, `zg6b2.py fit` for scr-g6b2- names, and
  `zr2t.py fit` for scr-r2t- names, added in this commit (KG, section 4, amended). It refuses a name that no stage
  there fits, or a fit queued twice, before reading any data.
- **zkg.py is not changed.** Its take stops on a scr-r2t- folder, so it is never run on an R2t session's output.
  `zr2t.py take` takes only scr-r2t- folders.
- **The session.** One session of a KG kernel on KG's image, after U2n's and G6b2's sessions, at this commit:
  - GPU 0: `scr-r2t-zg6m-hp`;
  - GPU 1: `scr-r2t-zg6m`, `scr-r2t-zrckpb-hp`, `scr-r2t-zrckpb`.

  Where a fit runs decides nothing.
- **Time.** KG's bases took, train and read together: zrckp 1,251 s (L-musique) and 1,242 s (L-hotpotqa); zg6 2,715 s
  and 4,685 s. R2 allowed its arms the bases' times plus 15%: about 1,440 s for each zrckpb fit, and 3,120 s and
  5,390 s for zg6m's. GPU 0 then takes about 1.5 hours and GPU 1 about 1.7. With the datasets' check, the session
  should take about 2 hours, and about 3.2 T4-hours in all. It costs nothing (KG, section 8).
- **Commands.** In the kernel, the stub runs the session, and zkg_session runs
  `zr2t.py fit --name N --rows-cap 65536` for each R2t name. On the laptop, after the session:

      python outputs/mp_unified/zr2t.py take --from <the kernel's output folder>
      python outputs/mp_unified/zr2t.py compare-all
      python outputs/mp_unified/zr2t.py verdict

- **What is committed.**
  - Committed: R2t's compares (outputs/screen/scr-r2t-*) and outputs/r2t: the pairs, the re-calls and the verdict,
    which carries the beside rows, the EM term and the timing.
  - Never committed: the fits' folders, weights, reads, logs and session files.
- **GL part 2.** GL's R2 fits (`scr-gl-zrckpb`, `scr-gl-zg6m` and their `-hp` twins) on workspace 3 from 1 November,
  if they run, are a second device's reading of R2 and decide nothing. zgl2.py is not changed, and its R2 verdict files
  are then read beside R2t's (GL, section 3, amended). G6's twin there is unchanged.

## 8. Self-tests, passed before this commit

`python outputs/mp_unified/zr2t.py --selftest` runs on fake files in a temporary folder:
1. **Names.** Four fits, four compares (zg6m's against zg6 first, zspk beside) and three calls. R2's arms are the ones
   in the arm table, R2's constants are as declared (59,537 parameters), and R2g and R2e are the next forms.
2. **Where.** train and read refuse anything but a Kaggle T4 on KG's image, and any name that is not R2t's, before
   reading data. fit refuses another rows cap.
3. **Train**, on fakes of rmatch's train:
   - another arm under R2's name, or other R2 constants, stops before any data;
   - each fit trains inside R2's EM term (the listwise loss, the head's inputs and the epoch hook all R2's), on the
     base build's roots, with U2d's card and seed 0;
   - the stamp holds the device, the rows cap, the bases, the code's sha256s, R2's constants and the term for each
     epoch;
   - every hook and root is restored after;
   - a fit whose batches carried no term, or whose train failed, stays unstamped.
4. **Read**, on fakes of rmatch's read: at 65,536 rows a batch on the evaluation carves, stamped. It refuses another
   device, a fit that is not R2t's, other R2 constants and a read that wrote another cap. A refused read runs nothing.
5. **fit** trains and reads once, keeps a stamped read, only reads a fit trained before, and writes r2t_timing.json.
6. **Stamps and take.**
   - It refuses an unstamped fit, and a read on another device or at another cap.
   - take copies records, reads and the timing file, never models.pt, and leaves KG's fits to KG.
   - take refuses an unread fit, a name that is not R2t's, and a file with other bytes.
   - zkg.py's take stops on an R2t folder.
7. **One device** across an R2t fit and its KG bases.
8. **The verdict**, each arm on its own files, with its beside rows (zg6m against zspk, R@5, the MLP shares, the term
   and the timing), and the pairs' and re-calls' stamps and text:
   - ADOPT_FOR_FULL_RUN;
   - NOT_ADOPTED with R2g next: a GAIN and a LOSS;
   - NOT_ADOPTED with R2e next: no GAIN; a LOSS alone; a GAIN that the null's raised floor turns WITHIN, through the
     move onto zg6.

   INCOMPLETE in ten cases, the other arm decided in each where it is not hit:
   - both arms: without KG's verdict file; with KG's verdict not BASE_READY; with the fits on another device than KG's;
   - zg6m only: without a zg6m compare; without KG's zg6-vs-zspk compare; with a zg6 column that is not KG's zg6 read;
   - zrckpb only: with a zrckp column that is not KG's zrckp read; without a KG zrckp null compare; with a compare that
     is not R2t's; with a read at another cap.
9. **compare-all** names and skips a compare whose fit is not taken (exit 1), keeps a filed one, and stamps each.
10. The loaders are restored, and nothing is written under R2t's real folders.

`python outputs/mp_unified/zkg_session.py --selftest` also checks that both R2t arms' names map to zr2t.py, that the
script exists, and that an R2t fit queued twice stops. `python outputs/mp_unified/zkg.py --selftest`,
`zg6b.py --selftest` and `zg6b2.py --selftest` pass unchanged.

The text encoder stays frozen. webqsp never trains. Test splits are never read. There is no LLM or transformer, and no
new graph.

## 9. Results

### 27 October 2026: zrckpb NOT_ADOPTED, zg6m NOT_ADOPTED

**Where it ran.** Section 7's session ran as the second version of kernel mpr-kg-g6b2, at f7b89d4, after G6b2's
output was downloaded.
- Its image was the one KG's fits name: every fit's device stamp, the session's Python (3.13.15), platform, GPUs (two
  T4s, 15,360 MiB) and driver equal KG's session 2.
- It first checked the nine datasets' 1,535 files (25.3 GB) against their manifests (164 s).
- GPU 0 ran scr-r2t-zg6m-hp (4,908 s). GPU 1 ran scr-r2t-zg6m (3,006 s), then -zrckpb-hp (1,485 s), then -zrckpb
  (1,411 s). All four exited 0; the session took 6,066 s.
- Where a fit runs decides nothing (section 7).

On the laptop at 578d72e (zr2t.py, zr2br.py, zkg.py and zg6.py as at f7b89d4): `zr2t.py take` (40 files),
`compare-all` and `verdict`. Records:
- outputs/screen/scr-r2t-zrckpb-vs-zrckp.json and .md, scr-r2t-zg6m-vs-zg6.json and .md, and their -hp twins;
- outputs/r2t: r2t-zrckpb-pair, r2t-zg6m-pair and r2t-zg6m-vs-zspk-pair, each with its -recall, and screen-verdict
  (.json and .md; 2026-10-27T10:18:35Z).

**zrckpb** ("zrckp + R2's bridge head") against KG's zrckp, re-called under KG's zrckp null:
- **1 GAIN:** L-musique musique (zero-shot), +0.0091 [+0.0035, +0.0149], floor 0.0075.
- **1 LOSS:** L-hotpotqa musique (in domain), -0.0133 [-0.0199, -0.0068], floor 0.0127.
- **One call changed** with the re-call: L-musique webqsp (+0.0120 [+0.0014, +0.0227], floor 0.0184) went from GAIN
  to WITHIN.

**zg6m** (zg6 + R2's bridge head on its final score) against KG's zg6, re-called under KG's zspk null moved onto zg6
through scr-kg-zg6-vs-zspk[-hp]:
- **0 GAIN.**
- **2 LOSS,** both webqsp (zero-shot): L-musique -0.0216 [-0.0393, -0.0047], floor 0.0075; L-hotpotqa -0.0605
  [-0.0778, -0.0424], floor 0.0242.
- **One call changed** with the re-call: L-musique musique (+0.0107 [+0.0053, +0.0163], floor 0.0138) went from GAIN
  to WITHIN.

**Next forms.** Section 6 fixed them before any number:
- zrckpb, NOT_ADOPTED with a GAIN and a LOSS: **R2g**, a learned per-question back-off on R2's head (the head times
  2 sigmoid(a . c_q + b), c_q the question's own embedding projection and its first-stage top-5 RRF margins, the gate
  starting at 1). One declared file.
- zg6m, NOT_ADOPTED with no GAIN: **R2e**, the edge form (bridge scores only for bridges linked to the row by zlink's
  edges, as fixed per-row link flags for the MLP and inside zg6's messages for the GNN). One declared file.
- Neither idea is closed (docs/SCREENS.md, section 3b).

Beside, deciding nothing:
- **zg6m against KG's zspk,** under zspk's null: 3 GAIN (L-musique metaqa +0.0110, L-hotpotqa metaqa +0.0076,
  L-hotpotqa musique +0.0158) and 3 LOSS, all zero-shot (L-musique webqsp -0.0969, L-hotpotqa hotpotqa -0.0169,
  L-hotpotqa webqsp -0.0815). L-musique musique (+0.0098, floor 0.0138) and L-hotpotqa 2wiki (+0.0168, floor 0.0219)
  went from GAIN to WITHIN. zg6's own call against zspk on the T4 was 4 GAIN and 2 LOSS (KG, section 10): the head
  moves zg6 little, and the read it moves most is webqsp's.
- **The bridge term trained.** On both arms and both splits it reached 99.4 to 99.5% of the training questions, and
  its loss fell every epoch: from 4.60 to 3.83 on L-musique and from 5.21 to 4.14 on L-hotpotqa. So the head learned
  its EM target; what it added to the ranking is the small moves above.
- **Where each arm moved.**
  - zrckpb raised both of L-musique's zero-shot reads (musique +0.0091, webqsp +0.0120) and both webqsp reads.
    It lowered L-hotpotqa's in-domain musique read (-0.0133) and its zero-shot hotpotqa read (-0.0050, WITHIN).
  - zg6m lowered both webqsp reads, by more on L-hotpotqa. Every other read is within 0.0107 of zg6, and every
    in-domain read within 0.0043.
- **The MLP share**, (zrckpb - rrf) / (zg6m - rrf), on the T4:
  - 0.97 to 0.99 on L-musique's metaqa, hotpotqa and 2wiki reads, and on L-hotpotqa's metaqa and 2wiki.
  - 0.74 on L-hotpotqa's musique read, where zrckpb has its LOSS.
  - 0.95 on L-musique's musique read (zero-shot), and 1.00 on its squad read, where the two are equal.
  - Above 1 where zrckpb is above zg6m: both webqsp reads, L-hotpotqa hotpotqa, and L-hotpotqa squad (1.20, where
    zg6m's gain over rrf is 0.0025).
- **Seconds and memory,** against KG's base of the same split on the same T4:

  | arm | split | train s (base) | read s (base) | peak GPU GB, train / read |
  |---|---|---|---|---|
  | zrckpb | L-musique | 1,107 (960), 1.15x | 298 (291) | 3.70 / 4.43, as zrckp's |
  | zrckpb | L-hotpotqa | 1,165 (963), 1.21x | 315 (279) | 5.55 / 5.55, as zrckp's |
  | zg6m | L-musique | 2,124 (1,942), 1.09x | 849 (773) | 4.66 / 5.24, as zg6's |
  | zg6m | L-hotpotqa | 4,166 (3,998), 1.04x | 714 (687) | 10.29 / 10.29, as zg6's |

  Peak RAM was 4.5 to 4.6 GB, as the bases'.
- zrckpb keeps the name "zrckp + R2's bridge head" until the user rules whether reading the five bridges' vectors keeps
  it inside the MLP track (section 2).
