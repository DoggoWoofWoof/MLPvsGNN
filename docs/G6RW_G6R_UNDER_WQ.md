# G6RW: G6r under WQ, on Kaggle's T4

Declared 28 October 2026, before any G6RW fit ran. G6r ([G6R_RELATION_MESSAGES.md](G6R_RELATION_MESSAGES.md)) was
declared on KG's splits and held on 27 October, before any fit, by its section 10. This file runs it under WQ
([WQ_WEBQSP_TRAINS.md](WQ_WEBQSP_TRAINS.md)), on WQK's base set ([WQK_WQ_ON_T4.md](WQK_WQ_ON_T4.md)). WQK's base verdict
(BASE_READY) and its floors are committed. Before this commit, WQK's base verdict was read only for its stage,
declaration and verdict, and WQK's G6 pair and re-call (`outputs/wqk/wqk-g6-zg6-pair*`) and the seed-0 R@5 in WQK's
floors on zspk were not opened. Code: `outputs/mp_unified/zg6rw.py` and its session, `outputs/mp_unified/zg6rw_session.py`.
zg6r.py, zu2n.py, zu2n_session.py, zwqk.py, zwqk_session.py and zwq.py are imported unchanged.

## 1. Why this, now

- **The program's next form.** PSMU closed the pool-softmax family for the MLP's input (3e50651,
  docs/PSMU_UNIFORM_LIFT_ON_T4.md, section 8). Its next list starts: "G6r under WQ on the T4". WQK's section 9 fixed it
  first: "G6r under WQ, on the T4. docs/G6R_RELATION_MESSAGES.md is declared on the old splits and held. Under WQ, it
  needs U2n's typed links for webqsp's fit carve, and its encoder from WQK's zspk seed-0 fits." This is that file.
- **Why G6r was held.** The user ruled on 27 October that webqsp trains and metaqa is read zero-shot. Under that
  protocol, G6r's relation map P learns from webqsp's 7,058 relations, and metaqa's nine are the unseen ones. That is
  the question G6r's first risk said KG's splits cannot ask (G6r, section 10):
  - every metaqa relation has a webqsp relation within cosine 0.60 to 0.90 of it;
  - only 1.6% of webqsp's relations have a metaqa relation within 0.6.
- **Why the T4.** The host is offline. Both of KG's Kaggle kernels are idle, and WQK's floors and seed-0 fits are on
  the T4. A WQ stage on the T4 takes WQK's seed-0 fits as its bases and WQK's floors as its floors (WQK, section 9).
- **The question.** Does a free GNN whose typed KB edges send relation-conditioned messages gain over WQK's zspk, on
  the in-domain reads and on the zero-shot ones (metaqa in both splits, and musique or hotpotqa)? That is Claim 2's
  model (an LLM-free GNN at SOTA), on the protocol the claims are now made on.

## 2. What changes, and what does not

| | G6r (held) | G6RW |
|---|---|---|
| model | zg6r (zg6r.py) | the same, imported unchanged: 180,218 parameters at hidden 128 |
| fits | `scr-g6r-zg6r[-hp]` | `scr-g6rw-zg6r`, `scr-g6rw-zg6r-hp` |
| splits | L-musique, L-hotpotqa (metaqa trains, webqsp never) | WQ-musique, WQ-hotpotqa (webqsp trains, metaqa zero-shot) |
| roots | U2d's base build | WQK's (zwqk.on_wqk, with WQK's mixed KT input) |
| typed links | U2n's metaqa/fit, metaqa/s1eval, webqsp/s1eval | U2n's metaqa/s1eval and webqsp/s1eval, and webqsp/fit's (section 3) |
| encoder's start | KG's seed-0 zspk fits | WQK's seed-0 zspk fits, `scr-wqk-zspk-s0[-hp]` (section 4) |
| when it trains | after KG's BASE_READY | after WQK's BASE_READY, committed before the stub |
| bases | KG's zspk (decides), KG's zg6 (beside) | WQK's seed-0 zspk (decides), WQK's seed-0 zg6 (beside) |
| floors | KG's zspk null | WQK's zspk seed null (`outputs/wqk/floors-zspk.json`) |
| MLP column | KG's zrckp | WQK's seed-0 zrckp |
| read batch | 32,768 rows | the same |
| record | `outputs/g6r/screen-verdict` | `outputs/g6rw/screen-verdict` |

The rule is G6's, and the next forms are G6r's section 6 under WQ (section 7). G6r's records, code and self-tests stay
as committed; nothing here writes under outputs/g6r.

## 3. Webqsp's fit carve's typed links

- **Why they are new.** G6r's carve reads U2n's typed links on every KB carve it loads. Under KG's splits webqsp never
  trained, so U2n built webqsp's links for s1eval only. Under WQ, webqsp's fit carve trains, so its typed links are
  needed. metaqa/fit is not read under WQ.
- **How they are built.** `zg6rw.py typed`, on the laptop, runs U2n's code (zu2n.build_typed, unchanged) on WQK's
  carve of webqsp/fit: WQK's P1c2 look (the look WQK's link records were built on), cache, links and rmatch chains,
  into `outputs/g6rw/typed/webqsp/fit`. Each part's cache and link records must be the ones WQK's ship.json names, or
  typed stops before writing typed.json. The records carry U2n's stage and declaration, since U2n's code writes them.
  `outputs/g6rw/typed.json` is G6RW's record of them.
- **Built on 28 October**, in 42.8 s; U2n's own check found the link arrays IDENTICAL to WQK's:
  - 2 parts, 16,329,970 edges;
  - 10,201,918 typed (every structural edge carries a relation; none without one);
  - 9,827 relation sets, over webqsp's 7,058 relations. Edges carry 1, 2, 3 or 4 relations: 6,761,002, 3,153,804,
    158,118 and 128,994 edges;
  - parts built on WQK's records (cache, link): part_0of2 (0bfbc52719d3, f509caa9151d), part_1of2 (ea3dbc0960f8,
    4d35babf8acb).
- **Direction bits** (input counts, no score):
  - webqsp/fit: 3,503,475 edges forward only, 3,503,475 back only, and 3,194,968 both (0.313 of the typed edges);
  - webqsp/s1eval (U2n's): 4,020,706 of 12,702,616 both (0.317);
  - metaqa/s1eval (U2n's): none of 120,133,110 both.
- **The s1eval carves stay U2n's.** U2n's typed records for metaqa/s1eval (13 parts) and webqsp/s1eval (2 parts) name
  the cache and link records of U2d's base build. WQK's overlay (W9) links those records byte-equal at WQK's roots.
  Checked on the laptop: all 26 and 4 records match. So zg6r's carve reads them under WQK as it did under KG.
- **What is committed:** the three typed records and typed.json. The arrays are not committed; they go to the
  session's dataset (section 8).

## 4. The encoder's start

- **The file.** Each split's encoder is WQK's seed-0 zspk fit of that split, `scr-wqk-zspk-s0` and
  `scr-wqk-zspk-s0-hp`: its models.pt (8,182,795 bytes each), kept on the laptop from WQK's second session's output,
  never committed. Their sha256 are pinned in zg6rw.py (ENC_SHA), and each is the one WQK's filed train.json of that
  fit names:
  - WQ-musique: 30cc29613f948f01a4de0aec4e58d6e31126ab028fb8cef18d7f7e69d62fcf09;
  - WQ-hotpotqa: 25e940c2fc47e049ac740b3b5cdd2e27798c188496eeb17447084e91fc4b720e.
- **ship.** `zg6rw.py ship --enc-from DIR` copies each to `outputs/g6rw/enc/<split>.pt` and loads it as the kernel
  does. It checks the typed arrays against their records and writes `outputs/g6rw/ship.json`: 7 files, 65,513,372
  bytes (the 2 encoders, sets.npy, and each part's t_set and t_dir).
- **The copy.** The 15 tensors zspk and zg6 share, from p@swa: zrm's block encoder (U, V), its chain match (the nine
  cm_ tensors) and its two encoder layers (l1, l2). That is 224,737 numbers at hidden 128. They are copied into each zg6r
  built while the encoder is active (zg6r's own copy, unchanged), then trained with everything else. The load refuses:
  - a file whose sha256 is not its pin;
  - a file that is not that fit at seed 0 with a p@swa;
  - a missing tensor, or another number of elements.

  A fit not trained from the encoder (no copy made, or another file) is INCOMPLETE (section 5).

## 5. The fits, the device, the compares and the rule

| G6RW fit | split | its bases, all on the same T4 |
|---|---|---|
| `scr-g6rw-zg6r` | WQ-musique | `scr-wqk-zspk-s0` (decides); `scr-wqk-zg6-s0` (beside) |
| `scr-g6rw-zg6r-hp` | WQ-hotpotqa | `scr-wqk-zspk-s0-hp` (decides); `scr-wqk-zg6-s0-hp` (beside) |

- **The same path as WQK's zg6.** zg6r is trained by zrm's train and read by zrm's read, at WQK's names and roots,
  with WQK's mixed KT input and P1c2's card, seed 0. G6RW's two inputs (sections 3 and 4) are active only inside the
  fit. Each fit reads p@swa on the six s1eval carves, 32,768 rows a batch (whole questions, so the scores do not
  change). fit refuses any other batch.
- **The stamps.** Each fit's screen.json carries:
  - WQK's wq block, at WQK's roots and declaration, naming the session's gate;
  - G6RW's block: the device, the gate and overlay, the encoder's record (file, sha256, fit, and how many zg6r were
    built from it), typed.json's sha256, the KT and typed-link records, zg6r's settings and its typed roots, and the
    code's sha256s.

  Its read.json is stamped with the device and the rows cap.
- **The device is KG's.** Train and read run only in a Kaggle kernel on a T4 from KG's image
  (kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3), checked before any data is read. A read must run on its fit's
  device. G6RW's fit of a split and WQK's seed-0 zspk and zg6 fits of that split must be on one device.
- **Before G6RW is called:**
  - WQK's base verdict (`outputs/wqk/base-verdict.json`) must be BASE_READY;
  - G6b2's verdict (`outputs/g6b2/screen-verdict.json`) must be NOT_ADOPTED with G6r next;
  - every fit's train, wq block and read must name a filed PASS gate of a G6RW session, and its overlay must be that
    gate's.
- **Two compares**, `scr-g6rw-zg6r-vs-zspk` and `-hp`. Each is lean_screen's compare (p@swa on s1eval) of the fit
  against two bases: WQK's seed-0 zspk first (it decides), then WQK's seed-0 zg6. Each is stamped with G6RW's record
  and the device.
- **The call.**
  1. screen_pair over the two compares against zspk: `outputs/g6rw/g6rw-zg6r-pair`.
  2. A re-call under WQK's zspk seed null: `outputs/g6rw/g6rw-zg6r-pair-recall`. WQK's floors are screen_recall's,
     max(0.0075, 2·sqrt((D1² + D2²)/2)), from WQK's seed-1 and seed-2 zspk fits. A read's base R@5 must equal the
     floors' seed-0 R@5 for that read.
- **The rule** is G6's (docs/G6_FREE_GNN.md, section 4), on the twelve reads (2 splits x 6 datasets):
  - **ADOPT_FOR_FULL_RUN:** no LOSS on any read and at least one GAIN;
  - **NOT_ADOPTED:** otherwise;
  - **INCOMPLETE:** any of these:
    - WQK's base verdict is not BASE_READY;
    - G6b2's verdict is not NOT_ADOPTED with G6r next;
    - a fit or compare is missing, or is not G6RW's;
    - G6RW's fits and WQK's seed-0 zspk and zg6 fits are not on one device;
    - a fit or read names a gate that is not a filed PASS gate of a G6RW session;
    - a fit was not trained from WQK's zspk encoder and typed.json's typed links;
    - WQK's floors are not their compares';
    - a re-call does not hold the twelve reads.

    The verdict names what is missing.

## 6. Beside, deciding nothing

- **zg6r against zg6**, the compares' second base, through the same pair and re-call
  (`outputs/g6rw/g6rw-zg6r-vs-zg6-pair` and `-recall`).
  - WQK fits no zg6 seed null, so the floors are zspk's, as a proxy.
  - zg6's seed-0 R@5 comes from WQK's own `scr-wqk-zg6-vs-zspk[-hp]` compares. Each G6RW compare's zg6 column must
    equal it.
  - This read says what the two changes together do over zg6.
- **R@5 per read on the T4**, for rrf, zspk, zg6, zg6r and zrckp (WQK's seed-0 fit, the MLP). The MLP share,
  (zrckp - rrf) / (zg6r - rrf), is shown where zg6r's gain over rrf exceeds 0.002.
- **The relation map** of each fit's p@swa, by zg6r's rel_stats (unchanged):
  - P_f's and P_b's norms and top singular values;
  - the share of each one's norm inside metaqa's centred relation span. Under WQ, that is the span of the KB no fit
    trains on;
  - each metaqa relation's |P_f r| and |P_b r|, and webqsp's top five by |P_f r|, with the mean share of webqsp's
    relations inside metaqa's span;
  - per layer, δ's mean, minimum and maximum, and the norms of A, B and C;
  - the encoder's drift from WQK's zspk encoder, |θ - θ_zspk| / |θ_zspk| over the 15 tensors.

  The kernel computes these from the fit's models.pt and writes g6rw_rel.json. The laptop never holds the weights.
- **Seconds and memory** for each fit, beside WQK's zg6 and zspk of the same split on the same T4.

## 7. What each outcome leads to (fixed now)

| zg6r's re-call under WQK's zspk null | what follows |
|---|---|
| ADOPT_FOR_FULL_RUN | zg6r under WQ fitted again on the pools WQP carried (U2s's pick), then its full run under WQ beside zspk's, then Claim 1 (P3c's intervals) and Claim 3 (cold latency, the C stages' timing) measured again; each its own declared file |
| NOT_ADOPTED, with a GAIN and a LOSS | zg6r under G6b's gate under WQ on the T4 (its edge pass and its no-edge pass mixed per question by the question's edge-family shares, as zg6b mixes zg6's; the encoder from WQK's zspk); one declared file |
| NOT_ADOPTED, with no GAIN | the two changes apart under WQ on the T4: the relation messages with zg6's own encoder (zg6r-m) and zg6 with WQK's zspk encoder start (zg6-z), each one fit pair; one declared file |
| INCOMPLETE | what is missing is supplied (WQK's base set, G6b2's verdict, a fit, a compare, a session's gate or a re-call); nothing else changes |

These are G6r's section 6 under WQ, with WQP's carried pools for U2s's: WQP kept U2s's pick under WQ
(docs/WQP_POOL_PICK_UNDER_WQ.md, section 9). No verdict here closes the free GNN (docs/SCREENS.md, section 3b). No model
change is adopted on today's pools (the user's rule of 13 October).

**The risks, stated before any number.**
- **metaqa's relations are unseen.** P learns from webqsp's 7,058 relations. metaqa's nine reach the codes only through
  their text vectors, and are read zero-shot in both splits.
- **Both directions.** Under KG's splits no both-way edge trained. Under WQ, webqsp's fit carve holds 3,194,968 of them
  (section 3), and metaqa's reads hold none. The both-way code is trained, but the zero-shot KB never uses it.
- **The encoder's start changes every dataset**, the passage graphs included, where no edge is typed. A change on a
  passage graph's read comes from the start, not the messages. Only the NO_GAIN form separates the two.
- **Not at equal training.** zg6r's encoder has trained on the split's questions twice: zspk's epochs, then zg6r's own.
  The deciding base is the encoder's own source fit.
- **The start.** A typed message starts as A_k B_k h_u, a rank-16 map, not zg6's W_0 h_u. zg6r starts as zg6 would with
  the same encoder weights only where no edge is typed.
- **Memory.** Under WQK on the T4, zg6 peaked at 2.28 GB training and 5.24 GB reading on WQ-musique, and at 9.13 GB
  training and reading on WQ-hotpotqa. zg6r's layers are checkpointed in training and its reads take 32,768 rows, as
  zg6e's were. Under KG, zg6e peaked at 4.11 / 4.67 GB and 8.21 GB, against zg6's 4.66 / 5.24 GB and 10.29 GB (G6r,
  section 3). The T4 has 15 GB. If a fit runs out of memory, the session records its exit code; a smaller batch is then
  an amendment declared before its rerun.
- **First run on Kaggle.** zg6r has passed its self-tests on the laptop's CPU but has never run in a kernel. A fit
  that does not exit 0 is named in session.json and makes the verdict INCOMPLETE.

## 8. How it runs

**One session**, a new version of whichever of KG's two kernels is free (mpr-kg or mpr-kg-g6b2). That keeps KG's
pinned image.
- **Datasets** (private, on the user's Kaggle account):
  - KG's nine;
  - WQK's three groups (mpr-wqk-1, -2, -3);
  - U2n's mpr-u2n, unchanged since U2n's session: the typed links' arrays of metaqa/fit, metaqa/s1eval and
    webqsp/s1eval, and tau0's;
  - **mpr-g6rw**, new: ship.json's 7 files at their repository paths, with a manifest.json (group g6rw);
  - the code bundle.
- **The checks** (zg6rw_session.py, standard library only):
  - KG's nine and U2n's arrays as U2n's session checks them: each array the one U2n's git-held records name;
  - WQK's groups as WQK's session checks them: exactly WQK's ship.json's files;
  - G6RW's group: exactly G6RW's ship.json's files, each at its size and sha256. ship.json must be built on
    typed.json as it is, and name only the two pinned encoders and the typed arrays G6RW's git-held records name;
  - no repository path in two datasets.

  Every file's size and sha256 is checked, and each is linked at its repository path.
- **Before anything runs:** WQK's verdict must be BASE_READY under WQK's declaration, typed.json must be G6RW's, and
  the clone must be its commit.
- **Before any fit:** WQK's overlay (W9) and gate (W10), each in its own process (`zwqk.py overlay`, `zwqk.py gate`).
  An overlay or gate that does not exit 0 stops the session.
- **The queues:**
  - GPU 0: `scr-g6rw-zg6r-hp`;
  - GPU 1: `scr-g6rw-zg6r`.

  No fit starts after 8 hours. Where a fit runs decides nothing.
- **Each fit**, in its own process: `zg6rw.py fit --name N --rows-cap 32768`. It refuses a name not G6RW's, another rows
  cap, a device that is not a T4 on KG's image, and a WQK verdict that is not BASE_READY. It trains, stamps and reads
  the fit, then computes its relation map. A read G6RW stamped is kept; a fit trained and stamped but not read is only
  read.
- **Time.** Under WQK, zg6 trained in 1,364.5 s and 3,355.6 s and read in 706.3 s and 715.0 s. zg6e, the nearest form,
  took about 1.5 times zg6's training. So zg6r should take about 2,050 s and 5,000 s to train and about 1,100 s to read
  each. The session should end within about 2 hours of the datasets' check. It costs nothing (the week's Kaggle GPU
  hours).

**On the laptop, after the session:**

    python outputs/mp_unified/zg6rw.py take --from <the kernel's output folder>
    python outputs/mp_unified/zg6rw.py compare-all
    python outputs/mp_unified/zg6rw.py verdict

- **take** copies each G6RW fit's records, reads, relation map and timing into `outputs/screen/fits/<name>`, never
  models.pt. It files the session's gate.json and overlay.json under `outputs/g6rw/sessions/<the gate's
  sha256[:12]>/`. A fit without a G6RW read on the device it was fitted on is left where it is, and named. take stops on
  a folder that is not one of G6RW's fits, a file already there with other bytes, or an output without the session's
  gate.
- **compare-all** runs lean_screen's compare of each fit against its split's WQK seed-0 zspk and zg6 fits. It requires
  all three on one device and stamps the compare with G6RW's record and that device. A compare G6RW stamped before is
  kept. Outputs: `outputs/screen/scr-g6rw-zg6r-vs-zspk[-hp].json`.
- **verdict** writes `outputs/g6rw/screen-verdict.json` and `.md`, with the pairs and re-calls. It exits 0 on
  ADOPT_FOR_FULL_RUN or NOT_ADOPTED and 2 after writing an INCOMPLETE record.

**The stub** (`zg6rw_session.py stub`) refuses:
- a name or rows cap not G6RW's;
- a WQK verdict that is not BASE_READY;
- a ship.json that is not the pinned encoders and the typed arrays G6RW's records name;
- a working tree that differs from HEAD;
- a HEAD that is not main, that origin does not hold, or that does not track any of: G6RW's, zg6r's, U2n's and WQK's
  code (every module zg6rw's stamps hash); WQK's ship.json, records, verdict and floors on zspk; G6RW's and U2n's typed
  records; typed.json; ship.json; and this file.

**What is committed:**
- Committed: this file, the code, the typed records, typed.json, ship.json, the two compares, the pairs, the re-calls
  and the verdict.
- Never committed: the encoder files (`outputs/g6rw/enc`), the typed arrays (.npy), the fits' folders, weights, reads
  and logs, session.json, the datasets' manifests, and `outputs/g6rw/sessions`.

## 9. Self-tests, passed before this commit

`python outputs/mp_unified/zg6rw.py --selftest` runs on toy files in a temporary folder:
1. **The names.** G6RW's fits and compares sit on WQK's seed-0 zspk and zg6 fits. The MLP column is WQK's zrckp
   compare, and zg6's is WQK's zg6 compare. Every module the stamps hash exists, once. The encoder's tensor names are
   zg6r's.
2. **zg6r.** A zg6r whose read batch is not G6RW's is refused. Inside zwq.on_wq, zg6r trains on zwq's modules; a
   module that is not zwq's is named and refused.
3. **The typed links' roots.** webqsp's fit carve comes from G6RW's root and every other carve from U2n's (zg6r's carve
   asks U2n's root). Another root is refused. U2n's carves and loader are restored after, on a failure too.
4. **The typed records** a stamp lists: webqsp's fit carve's at G6RW's root, the KB carves' at U2n's, none for a passage
   carve. A missing record stops.
5. **The guards.** Before any data is read:
   - split_of and train refuse a name not G6RW's (WQK's zg6, G6r's zg6r);
   - train and read refuse a place that is not a Kaggle kernel, a P100, and another image;
   - read and fit refuse a rows cap of 0, none or 65,536; fit refuses the laptop;
   - WQK's verdict is refused when missing, INCOMPLETE, under WQ's declaration or WQ's stage;
   - typed.json is refused when missing, not G6RW's, pinning a record with other bytes, or pinning another carve.
6. **The encoder.** It is loaded with its sha256 pinned: the 15 tensors, 224,737 numbers. While active it is zg6r's
   encoder, and zg6r's state is restored after. A file that is missing, not pinned, not seed 0 with a p@swa, without a
   tensor, or with another number of elements is refused.
7. **The stamps.** A trained fit's screen.json carries WQK's wq block at the toy's roots and G6RW's block: typed.json,
   the encoder, the device, the records, the typed roots and carves, and the read batch. A fit without them is refused.
8. **typed and ship.** typed refuses a part built on other link records than WQK's ship.json names: typed.json is not
   written, and U2n's carves are restored. ship refuses:
   - before typed;
   - a missing encoder file;
   - a filed train.json that does not name the encoder;
   - an array changed after its record.

   Otherwise ship writes the 7 files. The relation map is zg6r's rel_stats on the fit's p@swa, with the encoder.
9. **take, compare-all and the verdict**, on zwqk's toy base set (its fits, compares and floors) and a G6RW session:
   - **take** copies everything but models.pt, files the gate, and leaves an unread fit, naming it. It refuses other
     bytes, a folder not G6RW's, and an output without wqk/gate.json.
   - **compare-all** runs each compare with both bases, stamps it, and keeps it on a second run.
   - **The verdict**, each outcome with its beside rows:
     - ADOPT_FOR_FULL_RUN: the zg6 re-call, R@5 with zrckp's, the MLP share, the timing, both re-calls stamped, the
       markdown;
     - NOT_ADOPTED with a GAIN and a LOSS;
     - NOT_ADOPTED with no GAIN: a filed GAIN that a raised floor turns WITHIN.

     INCOMPLETE for each of: a missing compare; WQK's verdict missing; G6b2's verdict missing, or with another next
     form; a fit on another device; a gate not filed; a fit not trained from WQK's encoder or on typed.json's links; a
     read with another rows cap; a FAIL session; floors that are not their compares'.
10. Nothing is left patched in zu2n or zg6r, and nothing is written under G6RW's real folders.

`python outputs/mp_unified/zg6rw_session.py --selftest` runs on a toy repository and toy datasets:
1. **find and plan.** KG's nine groups and U2n's, WQK's two and G6RW's are found, and every file is planned once. The
   real pins refuse the toy's encoders.
2. **The refusals:**
   - in G6RW's manifest: a path outside ship.json (another split's encoder, a U2n path), a ship file in no manifest, a
     file listed twice, other bytes or size, no files;
   - in ship.json: not G6RW's, empty, not built on typed.json as it is, naming other bytes, missing an array, naming an
     extra file, or missing;
   - a typed part record or carve record changed after the record that names it;
   - a repository path in two datasets;
   - G6RW's group missing or attached twice.
3. **The run on the toy.**
   - G6RW's encoders and typed arrays, and U2n's arrays, are linked at their repository paths.
   - The overlay, then the gate, then both fits run, each with `--rows-cap 32768`.
   - session.json names this file, typed.json, both ship.json files and WQK's verdict.
4. **A failing overlay or gate** stops the session before any fit.
5. **Refused before anything runs:**
   - WQK's verdict not BASE_READY, or missing;
   - typed.json not G6RW's, or missing;
   - a clone that differs from its commit;
   - the real pins;
   - a name that is not G6RW's (G6r's, WQK's, a compare's), or a fit queued twice;
   - a rows cap of 65,536 or 0.
6. **The stub.** Its code compiles and runs G6RW's session. The stub refuses:
   - a name or rows cap not G6RW's;
   - WQK's verdict not BASE_READY;
   - the real pins;
   - an untracked G6RW part record, U2n record or zg6r.py;
   - a dirty tree;
   - a HEAD that is not main or that origin does not hold.

   With all of them held, it writes the verified bundle and the stub.
7. **The constants:** G6RW's two fits, and every module zg6rw's stamps hash, tracked.

The text encoder stays frozen. Test splits are never read. There is no LLM or transformer, no topic-entity help and no
new graph. webqsp trains under WQ's ruling only.

## 10. Results

**zg6r: ADOPT_FOR_FULL_RUN** (outputs/g6rw/screen-verdict.json and .md). Against WQK's seed-0 zspk, re-called under
WQK's zspk seed null: 5 GAIN and 0 LOSS on the twelve reads.

**The session.**
- Version 9 of mpr-kg at e727dc8 on 2x Tesla T4: 5,475.7 s.
- Overlay exit 0; gate exit 0 (fd5c2834b389).
- scr-g6rw-zg6r: GPU 1, exit 0, 2,941.8 s. scr-g6rw-zg6r-hp: GPU 0, exit 0, 4,946.8 s.
- Taken by `zg6rw.py take` (2 fits, 24 files), compared by `compare-all` and called by `verdict` (exit 0).

**The twelve reads.** R@5 on the T4. The call column is under WQK's zspk floors. The MLP share is
(zrckp − rrf) / (zg6r − rrf), from WQK's seed-0 zrckp.

| read | rrf | zspk | zg6 | zg6r | zg6r − zspk | floor | call | vs zg6 | MLP share |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| WQ-musique metaqa (zero-shot) | 0.0047 | 0.3984 | 0.4393 | 0.4105 | +0.0121 | 0.0439 | WITHIN | WITHIN | 1.0710 |
| WQ-musique squad | 0.9054 | 0.9110 | 0.9097 | 0.9105 | −0.0005 | 0.0075 | WITHIN | WITHIN | 1.0196 |
| WQ-musique musique (zero-shot) | 0.4734 | 0.4590 | 0.5018 | 0.4106 | −0.0484 | 0.0545 | WITHIN | LOSS | n/a |
| WQ-musique hotpotqa | 0.6846 | 0.9058 | 0.9109 | 0.9134 | +0.0076 | 0.0075 | **GAIN** | WITHIN | 0.9637 |
| WQ-musique 2wiki | 0.6079 | 0.9345 | 0.9390 | 0.9369 | +0.0024 | 0.0075 | WITHIN | WITHIN | 0.9836 |
| WQ-musique webqsp | 0.0535 | 0.6334 | 0.6423 | 0.6501 | +0.0167 | 0.0174 | WITHIN | WITHIN | 0.9886 |
| WQ-hotpotqa metaqa (zero-shot) | 0.0047 | 0.3384 | 0.4167 | 0.3834 | +0.0450 | 0.0308 | **GAIN** | LOSS | 1.0246 |
| WQ-hotpotqa squad | 0.9054 | 0.9114 | 0.9114 | 0.9111 | −0.0003 | 0.0075 | WITHIN | WITHIN | 1.1053 |
| WQ-hotpotqa musique | 0.4734 | 0.5794 | 0.5961 | 0.5941 | +0.0147 | 0.0090 | **GAIN** | WITHIN | 0.8360 |
| WQ-hotpotqa hotpotqa (zero-shot) | 0.6846 | 0.8496 | 0.8183 | 0.8529 | +0.0033 | 0.0075 | WITHIN | GAIN | 0.9976 |
| WQ-hotpotqa 2wiki | 0.6079 | 0.9330 | 0.9414 | 0.9420 | +0.0090 | 0.0075 | **GAIN** | WITHIN | 0.9638 |
| WQ-hotpotqa webqsp | 0.0535 | 0.6153 | 0.6421 | 0.6536 | +0.0383 | 0.0089 | **GAIN** | WITHIN | 0.9743 |

**What changed G6's call.** G6 under WQ would be NOT_ADOPTED (WQK, section 11). Its one LOSS was WQ-hotpotqa's
zero-shot hotpotqa read: −0.0313. zg6r reads it at 0.8529, +0.0033 over zspk, WITHIN. zg6r keeps zg6's four GAINs:
WQ-hotpotqa's metaqa, musique, 2wiki and webqsp. It adds WQ-musique hotpotqa, at +0.0076 against a floor of 0.0075.

**Beside, deciding nothing: zg6r against WQK's seed-0 zg6 under zspk's floors.** 1 GAIN and 2 LOSS.
- GAIN: WQ-hotpotqa hotpotqa (zero-shot), +0.0346.
- LOSS: WQ-musique musique (zero-shot), −0.0912.
- LOSS: WQ-hotpotqa metaqa (zero-shot), −0.0333.

All three moves are on held-out reads. On the eight in-domain reads zg6r is WITHIN zg6.

**What the risks of section 7 showed.**
- **WQ-musique's zero-shot musique read is the weak one.**
  - zg6r reads 0.4106. That is −0.0484 against zspk, 0.0061 inside its floor of 0.0545.
  - It is also −0.0628 below rrf (0.4734). zspk is below rrf too, at 0.4590.
  - The adoption is by the rule. The full run's intervals decide this read.
- **The encoder's start.** The three passage-graph GAINs (WQ-musique hotpotqa, WQ-hotpotqa musique and 2wiki) are on
  graphs where no edge is typed. zg6 reads each within 0.0025 of zg6r (0.9109, 0.5961, 0.9414). So they come from
  message passing and the start, not from the relation messages. Only the NO_GAIN arm (zg6-z) would separate the start
  from zg6's own fit, and it does not run on this verdict.
- **Not at equal training.** zg6r's encoder trained twice on each split's questions. The full run's declared file
  states how its base is matched.
- **metaqa's relations are unseen.**
  - The share of P's norm in metaqa's centred relation span: 0.2252 and 0.2165 (P_f, P_b) on WQ-musique; 0.2239 and
    0.3096 on WQ-hotpotqa.
  - webqsp's 7,058 relations sit 0.0427 inside that span on average.
  - metaqa's nine relations still get distinct codes, |P r| 0.52 to 2.18.
  - zg6r's zero-shot metaqa reads are +0.0121 and +0.0450 over zspk, and −0.0288 and −0.0333 under zg6.
- **Memory.** Peak GPU memory was below zg6's on both splits. The 15 GB T4 was not the limit.
  - WQ-musique: train 2.05 GB, read 4.59 GB (zg6 2.28 / 5.24).
  - WQ-hotpotqa: 6.01 / 6.01 GB (zg6 9.13 / 9.13).
  - Peak RAM was 4.56 and 4.67 GB.
- **Seconds.** zg6r took 1.33x and 1.18x zg6's training seconds (1,819 s and 3,958 s against 1,364 s and 3,356 s). Its
  reads took 1.54x and 1.35x (1,088 s and 962 s against 706 s and 715 s): they read 32,768 rows a batch.
- **First run on Kaggle.** Both fits exited 0.

**The MLP share, beside.** These are seed-0 points, not P3c's intervals, so they claim nothing.
- WQK's zrckp keeps 0.96 to 1.11 of zg6r's R@5 gain over rrf on ten of the eleven reads where the gain exceeds 0.002.
  On squad that gain is 0.005 to 0.006, so its shares (1.0196, 1.1053) say little.
- The exception is WQ-hotpotqa's in-domain musique read: 0.8360.
- On WQ-musique's zero-shot musique read zg6r is below rrf, so there is no share.

**Next** (section 7, fixed before any number). zg6r under WQ is fitted again on the pools WQP carried (U2s's pick). Then
comes its full run under WQ beside zspk's. Then Claim 1 (P3c's intervals) and Claim 3 (cold latency, the C stages'
timing) are measured again. Each is its own declared file. The refit needs U2s's pick built under WQ on WQK's carves,
which is its own declared stage. No model change is adopted on today's pools (the user's rule of 13 October).
