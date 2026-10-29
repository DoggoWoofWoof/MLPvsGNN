# R2WS: R2's bridge head (zrckpb) under WQ at the unseen seeds, on Kaggle's T4

Declared on 29 October 2026, before any R2WS fit. The code is `outputs/mp_unified/zr2ws.py` (the fits, the take, the
compares and the verdict) and `outputs/mp_unified/zr2ws_session.py` (the Kaggle session). They import WQK's code
(`zwqk.py`, `zwq.py`, `zwqk_session.py`, `zkg_session.py`), R2's (`zr2br.py`) and R2GW's (`zr2gw.py`) unchanged, and
patch them in process. No committed file is edited. Nothing here is a result.

## 1. Why this, now

- R2GW (docs/R2GW_R2G_UNDER_WQ.md) fitted zrckpb, R2's head on zrckp, at seed 0 on both splits as its beside base. Its
  filed re-call of zrckpb against WQK's seed-0 zrckp under WQK's zrckp floors read 3 GAIN and 0 LOSS on the twelve reads:
  WQ-musique webqsp (+0.0144 against a floor of 0.0075), WQ-hotpotqa metaqa (+0.0533 against 0.0489) and WQ-hotpotqa
  webqsp (+0.0106 against 0.0075). The other nine reads are WITHIN.
- **Those three reads were taken before this file was written.** They are R2GW's beside reads. They decided nothing in
  R2GW and they decide nothing here: this file does not count seed 0 in its rule (section 5).
- The three GAINs are small and near their floors (the metaqa one is 0.0044 above its floor, the two webqsp ones are
  0.0031 and 0.0069 above the 0.0075 minimum). One seed cannot say whether they are the arm or the seed. R2WS asks that
  and nothing else.
- **The seeds.** WQK filed zrckp under WQ at three seeds per split: `scr-wqk-zrckp-s0[-hp]`, `-s1[-hp]` and `-s2[-hp]`
  (`zwqk.FITS`; `zwq.NULL_SEEDS = (1, 2)`). Seeds 1 and 2 are the null seeds that made `outputs/wqk/floors-zrckp.json`.
  R2WS fits zrckpb at **seeds 1 and 2**, the seeds R2GW did not fit, on both splits. Each is compared with WQK's zrckp
  fit of the same seed.
- No new arm and no new idea. zrckpb here is R2GW's zrckpb, unchanged, at a different `--seed`.
- Under WQ, webqsp trains, metaqa is zero-shot, and one wiki dataset is held out per split.

## 2. The arm

**zrckpb**, exactly R2GW's (`zr2br.ZRCKPB`, registered in lean_screen's table as `zrckpb`): zrckp plus R2's bridge head
and R2's EM term (LAM 1), 59,537 numbers in R2's part, R2's last layer 0 so that it starts as zrckp bit for bit. Its
constants, parameter names, EM term and settings are zr2br's and are checked against R2's declaration before any fit is
stamped (`zr2gw.check_r2`). The seed enters as the fit's `--seed` (the base model's seed, and R2's own generator at
seed + 5502, `zr2br.r2_init`); nothing else differs from R2GW's seed-0 fits.

**Naming.** zrckpb is called "zrckp + R2's bridge head" until the user rules whether reading the five bridges' vectors
keeps it inside the MLP track (R2, section 2). No claim set rests on the head until that ruling, whatever this verdict
is. No claim sets zrckpb against the GNN as the MLP.

## 3. The fits and the device

| fit | seed | split | trains on | zero-shot | compared with |
|---|---|---|---|---|---|
| scr-r2ws-zrckpb-s1 | 1 | WQ-musique | webqsp, squad, hotpotqa, 2wiki | metaqa, musique | scr-wqk-zrckp-s1 |
| scr-r2ws-zrckpb-s1-hp | 1 | WQ-hotpotqa | webqsp, squad, musique, 2wiki | metaqa, hotpotqa | scr-wqk-zrckp-s1-hp |
| scr-r2ws-zrckpb-s2 | 2 | WQ-musique | webqsp, squad, hotpotqa, 2wiki | metaqa, musique | scr-wqk-zrckp-s2 |
| scr-r2ws-zrckpb-s2-hp | 2 | WQ-hotpotqa | webqsp, squad, musique, 2wiki | metaqa, hotpotqa | scr-wqk-zrckp-s2-hp |

- Four fits. zp1c3p's train under WQ, as zwq trains zrckp, with the arm swapped in; WQK's roots and mixed KT input;
  R2's EM term on with LAM 1; `--seed` 1 or 2 by the fit's name. A fit with no training batch carrying R2's term stops
  unstamped. The seed is in the fit's screen.json, in WQK's `wq` block and in R2WS's block, and the verdict checks all
  three.
- Every read: p@swa on the six s1eval carves, `--rows-cap 65536` (WQK's), on the device the fit was trained on.
- Device: KG's image on a Tesla T4 (WQK's). R2WS's four fits and WQK's four zrckp fits they are compared with (seeds 1
  and 2, both splits) must share one device record.
- **Layout.** One kernel with two T4s, two fits per T4, each T4's queue running its fits one after the other. The two
  splits are spread over the T4s so that neither queue holds both long fits of one split:

      gpu0: scr-r2ws-zrckpb-s1, scr-r2ws-zrckpb-s2-hp
      gpu1: scr-r2ws-zrckpb-s1-hp, scr-r2ws-zrckpb-s2

  A seed-loop would not change this: each fit is its own process (`zr2ws.py fit --name N`), exactly as in R2GW's session,
  so the session's queue is the only layout. If the kernel shows one T4, all four queue on it.

## 4. Compares, null and rule

Compares: lean_screen's compare of p@swa on s1eval, on the laptop, one base each, stamped with R2WS's record and the
device. The compare is named for both fits:

- `scr-r2ws-zrckpb-s<k>-vs-zrckp-s<k>[-hp]`: zrckpb at seed k against WQK's zrckp at the same seed k and the same split.

Null: WQK's zrckp seed null (`outputs/wqk/floors-zrckp.json`), unchanged. Per read, seeds 1 and 2 of zrckp against its
seed 0; floor = max(0.0075, 2 x their root mean square). The floors file must equal its compares' floors, recomputed.
The same floors are applied to each seed's re-call, as R2GW applied them to seed 0. The re-call's "seed-0 R@5" (the base
the null is checked against, `screen_recall.recall_row`) is replaced, per seed, by the same-seed zrckp read in that
seed's own compare, as R2GW did for its zrckpbg-against-zrckpb call; the floors do not change. The compare's zrckp
column must equal WQK's compare `scr-wqk-zrckp-s<k>[-hp]`'s new column on all twelve reads, or the verdict is
INCOMPLETE.

Per seed, the twelve reads (two splits x six datasets) are called GAIN, LOSS or WITHIN by the same lean_screen compare
and `screen_recall` re-call as R2GW's.

**Rule, fixed now**, over the seeds fitted (1 and 2):

- ADOPT_FOR_FULL_RUN: at no seed is any of the twelve reads a LOSS, and at least one read is a GAIN at **every** seed
  fitted (the same read, a reproduced GAIN).
- NOT_ADOPTED: otherwise. This includes a LOSS at any seed, a seed with no GAIN, and GAINs that sit on different reads at
  different seeds.
- INCOMPLETE, when any of these holds:
  - WQK's base verdict is not BASE_READY;
  - a fit or a compare is missing, or is not R2WS's, or is of another seed or split than its name says;
  - R2WS's fits and the WQK zrckp fits they are compared with are not on one device;
  - a fit or a read names a gate that is not a filed PASS gate of an R2WS session;
  - a fit's R2 constants or EM term are not as declared, or its head names a gate (zrckpb is not gated);
  - a compare's zrckp column is not WQK's read of that seed;
  - WQK's floors are not their compares';
  - a re-call does not hold the twelve reads.

"Same read" is the strict reading of "a reproduced GAIN". The weaker form, a GAIN somewhere at each seed but on different
reads, is reported beside (section 5) and does not adopt.

## 5. Beside, deciding nothing

- R2GW's seed-0 call per read (from R2GW's filed re-call `outputs/r2gw/r2gw-zrckpb-pair-recall.json`, read as a file; n/a
  where it is absent), set in a row with seeds 1 and 2: the calls at the three seeds per read. Seed 0 is not in the rule.
- The weaker form: the number of seeds with at least one GAIN, and the reads that are GAIN at one seed only.
- R@5 per read for rrf, WQK's seed-0 zspk, zrckp at seeds 1 and 2, and zrckpb at seeds 1 and 2, with each seed's
  difference and the floor.
- R2's term per fit: the share of training questions with the term, and L_br at the first and last epoch.
- Seconds and memory per fit, beside WQK's zrckp of the same seed and split.

## 6. Outcomes, fixed now, and risks

| verdict | next form |
|---|---|
| ADOPT_FOR_FULL_RUN | zrckpb fitted again on PKO's pools (docs/PKO_PICK_O_POOLS_UNDER_WQ.md, outputs/pko), then its full run under WQ beside zrckp's, each in its own declared file. The user's naming ruling is still pending (section 2): no claim set rests on the head until they rule whether reading the five bridges' vectors keeps R2 inside the MLP track |
| NOT_ADOPTED | the seed-0 GAINs are called noise (they did not reproduce under this rule); zrckpb is not taken further. The R2 line goes on with R2e (R2EW, docs/R2EW_R2E_UNDER_WQ.md) only; R2e's GNN half stays its own later file. A verdict closes only its form |
| INCOMPLETE | what is missing is supplied (WQK's base set, a fit, a compare, a session's gate or a re-call); nothing else changes |

Risks:

- The floors are WQK's, from zrckp against itself at other seeds. A same-seed comparison of two arms is not the same
  noise as a seed-to-seed comparison of one arm; whether it is tighter or looser is not claimed here. The bar is the
  declared floor, as in R2GW.
- Seeds 1 and 2 are the two samples that made the floors, and their zrckp reads are the bases here. A read where zrckp at
  seed 1 happened to fall below seed 0 has both a raised floor and a lower base at that seed. This is the same data the
  floors were declared on and is not corrected for.
- Twelve reads at two seeds is 24 calls. A single GAIN at one seed on one read is expected sometimes by chance; the rule
  asks for the same read at both seeds and no LOSS anywhere for that reason, and it still rests on two seeds.
- Three GAINs at seed 0 sat close to their floors. A reproduced GAIN of a similar size may be a few thousandths and the
  rule does not ask for more than the floor.
- The naming ruling is pending (section 2).

## 7. How it runs

The stub refuses unless the tree is clean, HEAD is main, origin/main holds HEAD, WQK's verdict is BASE_READY, and HEAD
tracks every module the stamps hash, this document, R2's and R2GW's, and WQK's records, verdict and floors.

Datasets: KG's nine private datasets, WQK's groups, and the code bundle (a git bundle of main). No new dataset.

The session (`zr2ws_session.py run`) is R2EW's, with R2WS's names: it finds and checks the datasets and links them, runs
WQK's overlay and gate (each in its own process; if either does not exit 0, no fit runs), then the fits, one queue per
T4. No fit starts after 8 hours.

One kernel, two fits per T4 (the second WQ kernel once R2EW's session on it has ended):

    python outputs/mp_unified/zr2ws_session.py stub --out DIR --gpu0 scr-r2ws-zrckpb-s1,scr-r2ws-zrckpb-s2-hp --gpu1 scr-r2ws-zrckpb-s1-hp,scr-r2ws-zrckpb-s2 --rows-cap 65536

Time: about 290 s of setup (WQK's). R2GW's zrckpb fits on this image trained in 985-991 s and read in 300 s each, and
WQK's zrckp fits at seeds 1 and 2 trained in 873-928 s. So a fit takes about 1,290 s and two in sequence about 2,600 s;
the session takes about 0.8 h. Estimated from R2GW's seed-0 fits, not measured for R2WS.

On the laptop, after R2WS's session output is downloaded into its own new empty folder:

    python outputs/mp_unified/zr2ws.py take --from DIR
    python outputs/mp_unified/zr2ws.py compare-all
    python outputs/mp_unified/zr2ws.py verdict

Never committed: the per-fit folders under outputs/screen/fits, outputs/r2ws/sessions, and the session's logs and
session.json.

## 8. Self-tests

    PYTHONIOENCODING=utf-8 python outputs/mp_unified/zr2ws.py --selftest
    python outputs/mp_unified/zr2ws_session.py --selftest

Both run on toys in temporary folders and write nothing under the real folders. `zr2ws.py --selftest` covers:

- the names: the four fits, the four compares, WQK's seeds-1-and-2 zrckp fits as the bases, the seeds being WQK's null
  seeds, the modules the stamps hash, and that R2WS's arm is R2GW's zrckpb;
- the seed: `train` hands `--seed` k to zp1c3p and the stamps carry k; a fit whose record names another seed is refused;
- the guards: where, the read cap, WQK's verdict and gate, the device, R2's constants, and the kept fit;
- the stamps of the fits, the take, compare-all and the verdict on a toy of WQK's base set: ADOPT_FOR_FULL_RUN (the same
  read GAIN at both seeds), NOT_ADOPTED with a LOSS at either seed, with GAINs on different reads at the two seeds, with a
  seed that has no GAIN, and with a GAIN within a raised floor, and every INCOMPLETE case of section 4.

`zr2ws_session.py --selftest` covers the session's run on a toy (the steps first, then each queue, every fit with
`--rows-cap 65536`), a failing overlay or gate, the refusals, the stub's refusals and its files, and the tracked paths.

## 9. Results

**Verdict: zrckpb is NOT_ADOPTED**, called by `zr2ws.py verdict` and filed in `outputs/r2ws/screen-verdict.json` and
`.md`, with the pairs, the re-calls and `outputs/screen/scr-r2ws-zrckpb-s<k>-vs-zrckp-s<k>[-hp]`. No read is a GAIN at both
seeds, so the seed-0 GAINs of R2GW are called noise.

Session: mpr-kg version 11 (gate 8e0f3db91ebe, overlay 187.8 s, gate 84.5 s, 3,288.0 s in all) on 2x Tesla T4, image
kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3. All four fits exited 0 (1,357 to 1,408 s each).

Against WQK's zrckp of the same seed under WQK's zrckp floors:

- GAIN: seed 1: WQ-musique webqsp (+0.0134). Seed 2: WQ-musique metaqa (+0.0690) and WQ-hotpotqa webqsp (+0.0122).
- LOSS, at both seeds: WQ-hotpotqa musique (seed 1 -0.0175, seed 2 -0.0215) and WQ-hotpotqa hotpotqa (-0.0125, -0.0115).
  At seed 0 (R2GW) both of those reads were WITHIN; the head's cost on the WQ-hotpotqa split's musique and hotpotqa reads
  repeats at the two new seeds.
- Gains on different reads at different seeds (the weaker form, which does not adopt): seeds with at least one GAIN are
  1 and 2, the reads are WQ-musique webqsp, WQ-musique metaqa and WQ-hotpotqa webqsp, each at one seed only.
- Seconds on the T4: train 1,047 to 1,087 against 873 to 928 for WQK's zrckp; read 301 to 306 against 287 to 298; peak
  GPU 1.87 and 3.99 GB train, 4.43 GB read; peak RAM 4.57 to 4.66 GB. R2's EM term reached 0.991 and 0.992.

Next (section 6): zrckpb is not taken further and the R2 line goes on with R2e only. R2e's MLP half (R2EW, 2ac2fe0) was
filed NOT_ADOPTED before these fits were read.
