# R2GW: R2g under WQ on Kaggle's T4

Declared on 28 October 2026, before any R2GW fit. The code is `outputs/mp_unified/zr2gw.py` (the arm, the fits, the take,
the compares and the verdict) and `outputs/mp_unified/zr2gw_session.py` (the Kaggle session). Both import WQK's code
(`zwqk.py`, `zwq.py`, `zwqk_session.py`, `zkg_session.py`) and R2's (`zr2br.py`) unchanged. Nothing here is a result.

## 1. Why this, now

- R2t screened `zrckpb` (zrckp plus R2's bridge head and its EM term) against KG's `zrckp` on the T4 under KG's null
  (`outputs/mp_unified/zr2t.py`, filed 817dd75). It was NOT_ADOPTED with one GAIN and one LOSS: L-musique's zero-shot
  musique read rose by +0.0091 (floor 0.0075), and L-hotpotqa's in-domain musique read fell by -0.0133 (floor 0.0127).
- For a GAIN and a LOSS, R2's next form is R2g (`zr2br.py`, NEXT): a learned per-question back-off on R2's head, the
  head times 2 sigmoid(a . c_q + b), c_q the question's own embedding projection and its first-stage top-5 RRF margins,
  the gate starting at 1, in one declared file. This is that file.
- WQK's base set is BASE_READY on the T4 (9059de7). WQK's next forms (`zwqk.py`, NEXT) put R2g under WQ, in its own
  declared file, on WQK's seed-0 fits and WQK's floors.
- Under WQ, webqsp trains, metaqa is zero-shot, and one wiki dataset is held out per split: WQ-musique holds out
  musique and WQ-hotpotqa holds out hotpotqa.
- The pool-softmax family is closed (3e50651), WQP gave NO_GAIN (5352b17) and G6RW has filed (e0e2c51). Both Kaggle
  kernels, mpr-kg and mpr-kg-g6b2, are free.

## 2. What changes

One thing changes: R2's head is multiplied by a gate, one number per question.

Everything else is zrckpb as R2t ran it, now under WQ: zrckp's model and inputs, R2's five bridges, its head, its EM
term (LAM 1), seed 0, WQK's roots and WQK's mixed KT input. The gate scales only R2's term. zrckp's own score is not
touched. The gate lies between 0 and 2, so for a question it can switch the head off or at most double it.

## 3. The arms

**zrckpbg (R2g), deciding.** For row v of question q:

    score_v = s_v + g_q * h_v,    g_q = 2 * sigmoid(a . c_q + b)

s_v is zrckp's score and h_v is R2's head on the row (zrckpb's term). c_q holds 36 numbers:

- h_q = e_q @ r2_U, 32 numbers: the question projection R2 already learns for its bridge logits. The gate shares r2_U
  with R2 and does not detach it.
- m_q = (g_0, g_1, g_2, g_3), 4 numbers, with g_j = (r_j - r_(j+1)) / r_0. Here r_0 >= r_1 >= ... are the stored
  first-stage RRF scores (column c_rrf) of R2's five bridges: the question's top five rows by that column, descending,
  ties by pool position, missing values last (`zr2br.bridges_np`). g_j = 0 where bridge j+1 is absent or a value is not
  finite. m_q = 0 where r_0 <= 0 or r_0 is not finite. The margins are computed once per carve when it loads.

New parameters: a (36 numbers) and b (1), so 37 in all. R2's part of the model goes from 59,537 numbers (zrckpb) to
59,574 (zrckpbg).

The start: a = 0 and b = 0, so g_q = 2 * sigmoid(0) = 1 exactly, and 1 * h_v = h_v. zrckpbg therefore starts as zrckpb
bit for bit. R2's last layer starts at 0, so h_v = 0 and the gate's gradient is exactly 0 on the first step. The gate
starts to learn once the head has moved. R2's EM term reads only the bridge logits, so the gate leaves it unchanged.

**zrckpb, beside.** R2t's arm, unchanged, fitted under WQ next to zrckpbg. It shows what the gate does against its own
ungated twin, on the same device and inputs.

**Naming.** Both arms are called "zrckp + R2's bridge head" until the user rules whether reading the five bridges'
vectors keeps an arm inside the MLP track (R2, section 2). No claim sets either arm against the GNN as the MLP.

## 4. The fits and the device

| fit | arm | split | trains on | zero-shot |
|---|---|---|---|---|
| scr-r2gw-zrckpbg | zrckpbg | WQ-musique | webqsp, squad, hotpotqa, 2wiki | metaqa, musique |
| scr-r2gw-zrckpbg-hp | zrckpbg | WQ-hotpotqa | webqsp, squad, musique, 2wiki | metaqa, hotpotqa |
| scr-r2gw-zrckpb | zrckpb | WQ-musique | webqsp, squad, hotpotqa, 2wiki | metaqa, musique |
| scr-r2gw-zrckpb-hp | zrckpb | WQ-hotpotqa | webqsp, squad, musique, 2wiki | metaqa, hotpotqa |

- Every fit: seed 0; zp1c3p's train under WQ, as zwq trains zrckp, with the arm swapped in; WQK's roots and mixed KT
  input; R2's EM term on with LAM 1. A fit whose training had no batch with R2's term stops unstamped.
- Every read: p@swa on the six s1eval carves, `--rows-cap 65536` (WQK's), on the device the fit was trained on.
- zrckpbg's read also keeps its gate per question on each carve: `r2gw_gates.npz` in the fit's folder, and its count,
  mean, spread, quantiles and share below 1 in `read.json`.
- The device is KG's image on a Tesla T4 (WQK's). All four fits and WQK's seed-0 zrckp fit of each split must share
  one device record.

## 5. Compares, null and rule

Compares: lean_screen's compare of p@swa on s1eval, run on the laptop and stamped with R2GW's record and the device.

- `scr-r2gw-zrckpbg-vs-zrckp[-hp]`: zrckpbg against WQK's seed-0 zrckp of its split (the first base; it decides),
  then against zrckpb of the same split (beside).
- `scr-r2gw-zrckpb-vs-zrckp[-hp]`: zrckpb against WQK's seed-0 zrckp (beside).

Null: WQK's zrckp seed null (`outputs/wqk/floors-zrckp.json`). Per read, seeds 1 and 2 against seed 0; the floor is
max(0.0075, 2 x their root mean square). The floors file must equal its compares' floors, recomputed.

Rule, on zrckpbg's re-call against WQK's seed-0 zrckp under that null, over its twelve reads (two splits x six
datasets):

- ADOPT_FOR_FULL_RUN: no read is LOSS and at least one read is GAIN.
- NOT_ADOPTED: otherwise.
- INCOMPLETE, when any of these holds:
  - WQK's base verdict is not BASE_READY;
  - a fit or a compare is missing, or is not R2GW's;
  - R2GW's fits and WQK's seed-0 zrckp fits are not on one device;
  - a fit or a read names a gate that is not a filed PASS gate of an R2GW session;
  - a fit's R2 constants or EM term are not as declared (zr2br's constants, LAM 1, at least one question with the
    term);
  - zrckpbg's gate is not as declared in section 3, or its read holds no gate per question on the six carves;
  - WQK's floors are not their compares';
  - a re-call does not hold the twelve reads.

## 6. Beside, deciding nothing

- zrckpb against WQK's seed-0 zrckp under the same null: does R2's head repeat R2t's GAIN and LOSS under WQ?
- zrckpbg against zrckpb under the same floors, the null's seed-0 R@5 replaced by R2GW's zrckpb read: what the gate
  itself does.
- The gate per question on each read: mean, spread, quantiles and share below 1, and the share of questions whose
  margins are not all 0.
- R@5 per read for rrf, zrckp, zrckpb, zrckpbg and WQK's seed-0 zspk.
- R2's term per fit: the share of training questions with the term, and L_br at the first and the last epoch.
- Seconds and memory per fit, beside WQK's seed-0 zrckp of the same split.

## 7. Outcomes, fixed now, and risks

| verdict | next form |
|---|---|
| ADOPT_FOR_FULL_RUN | zrckpbg fitted again on the pools WQP carried (U2s's pick), then its full run under WQ beside zrckp's, each in its own declared file |
| NOT_ADOPTED | R2e under WQ on the T4, in its own declared file (R2's next form after R2g: bridge scores only for bridges linked to the row by zlink's edges) |
| INCOMPLETE | what is missing is supplied (WQK's base set, a fit, a compare, a session's gate or a re-call); nothing else changes |

A verdict closes only this form.

Risks:

- The margins assume the stored c_rrf column is the raw, positive RRF score. Where a carve's top value is 0 or
  negative, its margins are 0 and c_q falls back to h_q. The share of questions with non-zero margins is reported per
  read.
- The gate shares r2_U with R2's bridge logits. Once a is not 0, the gate's gradient also moves r2_U.
- The gate cannot learn before the head moves (its gradient is exactly 0 at the start). If the head stays near 0,
  zrckpbg stays near zrckpb.
- The gate caps R2's term at twice the head.
- The naming ruling is pending (section 3).
- Memory at 65,536 rows a read batch: R2's bridge logits are made in slices of 65,536 rows, as in R2t. The gate adds
  37 parameters and 4 floats per question.

## 8. How it runs

The stub refuses unless the tree is clean, HEAD is main, origin/main holds HEAD, WQK's verdict is BASE_READY, and HEAD
tracks every module the stamps hash, this document, and WQK's records, verdict and floors.

Datasets: KG's nine private datasets, WQK's groups, and the code bundle (a git bundle of main). No new dataset.

The session (`zr2gw_session.py run`) finds and checks the datasets and links them (zkg_session and zwqk_session,
unchanged). It runs WQK's overlay and gate, each in its own process; if either does not exit 0, no fit runs. Then it
runs the fits, one queue per T4, each fit in its own process. No fit starts after 8 hours. session.json records the
commit, the manifests, the versions, and each step's and fit's exit code and seconds.

Layout. Both kernels are free, so the four fits split two per kernel, one fit per T4:

    # mpr-kg-g6b2: the deciding pair
    python outputs/mp_unified/zr2gw_session.py stub --out DIR_A --gpu0 scr-r2gw-zrckpbg-hp --gpu1 scr-r2gw-zrckpbg --rows-cap 65536
    # mpr-kg: the beside pair
    python outputs/mp_unified/zr2gw_session.py stub --out DIR_B --gpu0 scr-r2gw-zrckpb --gpu1 scr-r2gw-zrckpb-hp --rows-cap 65536

With one kernel only, two fits per T4:

    python outputs/mp_unified/zr2gw_session.py stub --out DIR --gpu0 scr-r2gw-zrckpbg-hp,scr-r2gw-zrckpb --gpu1 scr-r2gw-zrckpbg,scr-r2gw-zrckpb-hp --rows-cap 65536

Both kernels run KG's image on a T4, so the fits share one device record. Each session files its own gate. The verdict
accepts a fit from any filed PASS gate of an R2GW session.

Time: about 290 s of setup per session (WQK's). On the T4, WQK's zrckp trained in 831-837 s and read in 283-297 s.
R2t's zrckpb fits took 1,411-1,485 s on KG's T4, and WQK's fits train in 0.70-0.95 of KG's seconds. So one zrckpb or
zrckpbg fit takes about 1,000-1,450 s. With two kernels, each session takes about 0.35-0.5 h, and they run at the same
time. With one kernel, the session takes about 0.65-0.9 h. Both are far under the 8-hour cut-off.

On the laptop, after `kaggle kernels output` of each session into its own new empty folder:

    python outputs/mp_unified/zr2gw.py take --from DIR_A
    python outputs/mp_unified/zr2gw.py take --from DIR_B
    python outputs/mp_unified/zr2gw.py compare-all
    python outputs/mp_unified/zr2gw.py verdict

Never committed: the per-fit folders under outputs/screen/fits (with `r2gw_gates.npz`), outputs/r2gw/sessions, and the
session's logs and session.json.

## 9. Self-tests

    PYTHONIOENCODING=utf-8 python outputs/mp_unified/zr2gw.py --selftest
    python outputs/mp_unified/zr2gw_session.py --selftest

Both run on toys in temporary folders and write nothing under the real folders. `zr2gw.py --selftest` covers:

- the names: the four fits, the four compares on WQK's seed-0 zrckp, WQK's zspk compare, the modules the stamps hash,
  R2's NEXT texts and R2t's naming text;
- zr2br checked as R2GW's (its constants, parameters and splits), another value refused;
- the margins by hand (ties, a missing value, a single row, an empty pool, a negative top) and their guards;
- the arm's classes and lean_screen's table;
- on a toy model: the gate is 1 and zrckpbg equals zrckpb bit for bit, in eval and in training; c_q by hand; 37 gate
  numbers and 59,574 for R2's part; with a and b moved, the score is s + g_q h; no R2 input leaves the base score; a
  batch without margins is refused;
- R2's EM term: the same bridge logits kept with and without the gate, a batch kept twice refused, the gate's gradient
  exactly 0 at the start and not 0 once the head has moved, and a fit with no term refused;
- the carve: R2's carve with each question's margins in its batches;
- the gate per question in a read: only p@swa's gates kept, in question order, the patch removed after, on an error
  too, a read without a gated p@swa refused, and no patch for zrckpb;
- zwq.on_wq: zr2br trains on zwq's modules, and the arm swap is restored;
- the guards: where, the read cap, WQK's verdict and gate, the device, R2's constants, and the kept fit;
- the stamps of all four fits, and the take, compare-all and verdict on a toy of WQK's base set: ADOPT_FOR_FULL_RUN,
  NOT_ADOPTED with a GAIN and a LOSS, a GAIN within a raised floor, and every INCOMPLETE case of section 5.

`zr2gw_session.py --selftest` covers the session's run on a toy (the steps first, then each queue in order, every fit
with `--rows-cap 65536`), a failing overlay or gate, the refusals (WQK's verdict, a fit not R2GW's, a fit queued twice,
another read cap), the stub's refusals and its files, and the tracked paths.

## 10. Results

**Verdict: zrckpbg is NOT_ADOPTED** (1 GAIN, 1 LOSS), called by `zr2gw.py verdict` and filed in
`outputs/r2gw/screen-verdict.json` and `.md`, with the pairs, the re-calls and `outputs/screen/scr-r2gw-*-vs-zrckp[-hp]`.

Session: mpr-kg-g6b2 version 6 (the zrckpbg pair, gate 4b6f360be563, 1,653.7 s) and mpr-kg version 10 (the zrckpb pair,
gate 7576c1848176, 1,544.3 s), both on 2x Tesla T4, image kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3. All four
fits exited 0. The sessions came back in about 13 minutes of wall time after the start check; the fits' own seconds are
1,304 to 1,339 each.

Deciding, zrckpbg against WQK's seed-0 zrckp under WQK's zrckp floors:

- GAIN: WQ-hotpotqa webqsp (0.6382 -> 0.6492, +0.0110, floor 0.0075).
- LOSS: WQ-hotpotqa musique (0.5743 -> 0.5613, -0.0130, floor 0.0075).
- The two metaqa zero-shot reads, +0.0304 (WQ-musique) and +0.0204 (WQ-hotpotqa), are GAIN on the pair's own intervals and
  WITHIN under the zrckp floors (0.0621 and 0.0489). The other eight reads are WITHIN.
- The gate backed off where the first-stage signal is weak: mean gate 0.71 on metaqa (0.87 to 0.90 of questions below 1)
  and 1.30 to 1.36 on webqsp. It did not help: zrckpbg gains less than zrckpb on WQ-hotpotqa metaqa (0.4131 against 0.4459).

Beside, deciding nothing:

- zrckpb (R2's head, no gate) against WQK's seed-0 zrckp under the same floors: 3 GAIN (WQ-musique webqsp +0.0144,
  WQ-hotpotqa metaqa +0.0532, WQ-hotpotqa webqsp +0.0106) and 0 LOSS. This is one seed and it was read here, not in a
  file that declared it; it is not adopted on that.
- zrckpbg against zrckpb: 0 GAIN, 1 LOSS (WQ-hotpotqa musique).
- Seconds on the T4: train 1,020 / 1,026 for zrckpbg, 991 / 985 for zrckpb, 831 / 837 for WQK's zrckp; peak GPU 1.87 and
  3.99 GB train, 4.43 GB read, equal for the three; peak RAM 4.5 to 4.6 GB.
- R2's EM term reached 0.991 to 0.992 of training questions on all four fits.

Next (section 7, fixed before any number): R2e under WQ on the T4, its own declared file. Written and pushed as R2EW
(2d487a2, `docs/R2EW_R2E_UNDER_WQ.md`), launched on mpr-kg-g6b2 version 7. Because the ungated zrckpb read 3 GAIN and 0 LOSS
beside, a confirmation at the unseen null seeds (zrckpb at seeds 1 and 2 against zrckp's same seeds) is being declared as
its own file before it runs.
