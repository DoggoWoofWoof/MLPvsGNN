# G6b: zg6 with a per-question back-off to its twin, on Kaggle's T4

Declared 27 October 2026. Code: `outputs/mp_unified/zg6b.py`. Its fits run in KG's sessions (`zkg_session.py`) on KG's
device. No G6b fit has run. Nothing in this file reads a G6b number.

## 1. Why this, now

- **G6's call fixes this form.** GL filed G6 NOT_ADOPTED on 27 October (docs/GL_GPU_LINE_ON_L40S.md, section 7). zg6
  against zspk, re-called under U2d's zspk null, had 2 GAIN and 3 LOSS. Before any number, G6's section 5 fixed the next
  form for that outcome: "G6b, a per-question back-off: zg6's score mixed with its twin's through a learned gate on the
  question's edge-family shares, at equal weight at the start (one declared file)". This is that file.
- **Where zg6 loses.** All three LOSSes are zero-shot reads: L-musique webqsp (-0.0795), L-hotpotqa hotpotqa (-0.0203)
  and L-hotpotqa webqsp (-0.0584). zg6 is above zspk on 7 of its 8 in-domain reads and below it on 3 of its 4 zero-shot
  reads. It fits the training graphs better and transfers worse. A back-off lets a question lean on the twin, which
  scores each row from its own inputs, where the edges do not help.
- **zg6t's own fits are not needed.** GL (sections 4 and 7) says G6b needs zg6t's fits. It does not.
  - G6's twin is "zg6 with no edges reaching its layers: the same parameters and readout" (G6, section 2).
  - In zg6b the twin is exactly that: the same network run a second time with no edges, trained together with the edge
    pass through the mixed score.
  - A separately fitted zg6t has its own weights. Mixing two separately trained models' scores is not this form.
  - zg6t's fits stay where GL put them (section 4, on workspace 3 from 1 November), beside, deciding nothing for G6b.
- **The device is KG's T4.** The lab host is offline. Workspace 3 renews on 1 November, workspaces 4 and 5 are spent,
  and 6 to 10 need the user's payment method. KG (docs/KG_KAGGLE_T4_LINE.md) fits U2d's base set again on Kaggle's T4
  and names G6b as the first stage after it. G6b's fits run in KG's sessions, and its bases are KG's fits on the same
  device.
- **The risk, stated before any number.** The gate learns only from the training splits' questions, which are
  in-domain. If the edges help there, the gate may lean on the edges everywhere, zero-shot included, and give back zg6's
  zero-shot losses. The gate's values on each dataset (section 5) show where it leans. If it leans that way, G6b2
  (section 6) moves the back-off to each row.

## 2. The model

zg6b is zg6's network (G6, section 2), unchanged:
- the encoder, the boundary (the eight KT columns and rmatch's two chain features), the context, the three layers and
  the readout;
- the edge-family dropout in training (G2's fd25 rule), with the same draws from the module's own generator;
- the training: rmatch.py's train on P1c2's card, as zspk's and zg6's fits (the listwise loss, eight epochs, config
  2e-3:1e-4:0.1:8:2, p@swa, hidden 128, variant p), on P1c3's carves with the KT input.

What is new:
- **Two passes from one encoder pass and one context.** h0 and the context weights are computed once. The three layers
  and the readout then run twice:
  - **s_e**, over the question's pool edges that reach the layers: all of them at read, less the dropped families in
    training. This is zg6's score.
  - **s_t**, with no edges: the twin's score. Every mean and attention average is 0, so each row's s_t depends only on
    its own inputs.
- **The gate.** For each question, sh holds each family's share among its edges that reach the layers. The families are
  the structural, named-entity and kNN families, and a question without edges has sh = 0. The gate is
  g = sigmoid(a · sh + b). a (three weights) and b start at 0, so g = 0.5 at the start.
- **The score**, on each of the question's rows: g · s_e + (1 - g) · s_t. A question without edges has s_e = s_t, so its
  score is the twin's, whatever g is.
- **Parameters.** 121,806 of its own at hidden 128: zg6's 121,802 and the gate's 4. From seed 0, every weight starts
  where zg6's does, and the gate starts at 0.

The gate reads only the edges' family shares, as G6 fixed. It does not read the dataset, the split or any gold.

## 3. The fits and the device

Two fits on U2d's base build, seed 0:

| G6b fit | split | its bases, KG's fits on the same T4 |
| --- | --- | --- |
| `scr-g6b-zg6b` | L-musique | `scr-kg-base-zspk` (decides), `scr-kg-base-zg6` (beside) |
| `scr-g6b-zg6b-hp` | L-hotpotqa | `scr-kg-base-zspk-hp` (decides), `scr-kg-base-zg6-hp` (beside) |

- **The same path as KG's.** zg6b.py trains and reads through zu2d.py's train and read bodies, as zkg.py runs them:
  - U2d's base build's roots and gate, P1c2's card, and the seed from the name;
  - reads of p@swa on the six s1eval carves with `--rows-cap 65536` (G6's batch; whole questions, so the scores do not
    change).

  Its stamp records the device, the build, the pools, the KT, rmatch, zrc and zlink records, and the model's settings.
- **The device is KG's.** Train and read run only in a Kaggle kernel on a T4 from KG's image
  (kaggle-gpu:py3.13.15:torch2.11.0+cu128:numpy2.1.3), checked before any data is read. A read must run on its fit's
  device.
- **KG must be BASE_READY** (outputs/kg/base-verdict.json) before G6b is called. G6b's two fits must be on KG's device.
- **Memory.** zg6's peaks on the L40S were 4.7 and 10.3 GB in training and 13.3 GB in its reads; the T4 has 16.1 GB.
  The twin's pass carries no edge tensors, so zg6b should add little. Each fit's g6b_timing.json records its peaks.

## 4. The compares, the null and the rule

- **Two compares**, `scr-g6b-zg6b-vs-zspk` and `-hp`. Each is lean_screen's compare (p@swa on s1eval) of zg6b's fit
  against two bases: KG's zspk, which comes first and decides, and KG's zg6, beside. All three fits must be on one
  device. Each compare is stamped with G6b's record.
- **The call.**
  1. screen_pair over the two compares against zspk: `outputs/g6b/g6b-zg6b-pair`.
  2. A re-call under KG's zspk seed null: `outputs/g6b/g6b-zg6b-pair-recall`. KG's floors are screen_recall's,
     max(0.0075, 2·sqrt((D1² + D2²)/2)), from KG's seed-1 and seed-2 zspk fits.
- **The rule** is G6's (G6, section 4), on the twelve reads:
  - **ADOPT_FOR_FULL_RUN:** no LOSS on any read and at least one GAIN.
  - **NOT_ADOPTED:** otherwise.
  - **INCOMPLETE:** any of these:
    - KG's base verdict is not BASE_READY;
    - a fit or compare is missing, or is not G6b's;
    - the fits are not on KG's device;
    - the re-call does not hold the twelve reads.

    The verdict names what is missing.

## 5. Beside, deciding nothing

- **zg6b against zg6**, the compares' second base, through the same pair and re-call
  (`outputs/g6b/g6b-zg6b-vs-zg6-pair` and `-recall`).
  - KG fits no zg6 seed null, so the floors are zspk's, as a proxy.
  - zg6's seed-0 R@5 comes from KG's own `scr-kg-zg6-vs-zspk` compares. Each G6b compare's zg6 column must equal it; a
    mismatch makes the verdict INCOMPLETE.
- **R@5 per read on the T4** for rrf, zspk, zg6, zg6b and zrckp (KG's seed-0 fit, the MLP). The MLP share,
  (zrckp - rrf) / (zg6b - rrf), is shown where zg6b's gain over rrf exceeds 0.002.
- **The gate** of each fit's p@swa:
  - a and b;
  - its value for a question of one family alone, and for a question without edges;
  - over each dataset's s1eval questions: its mean, its 10th, 50th and 90th percentiles, and the share of questions
    without edges.

  The kernel computes these from zlink's arrays, each checked against its part's record, and writes g6b_gate.json. The
  laptop holds only the arrays' records.
- **Seconds and memory** for each fit, beside KG's zg6 of the same split on the same T4.

## 6. What each outcome leads to (fixed now)

| zg6b's re-call under KG's zspk null | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | zg6b fitted again on P1g-b's pools, then its full run on all six splits beside zspk's, then Claim 1 (P3c's intervals) and Claim 3 (cold latency, the C stages' timing) measured again; each its own declared file |
| NOT_ADOPTED, with at least one GAIN and a LOSS | G6b2, a per-row back-off: the gate also reads the row's own log1p in-degree per family over the edges that reach the layers, so a row the edges barely reach can lean on the twin (one declared file) |
| NOT_ADOPTED, with no GAIN | G6r (G6, section 5): relation-aware messages from text, the relation embeddings (`outputs/m3b/relations/metaqa_rel_embeddings.npy`, `webqsp_rel_embeddings.npy`) through one shared low-rank map in place of the per-family maps on the KB datasets, the encoder started from zspk's trained encoder, still unbounded with its own readout (one declared file) |
| INCOMPLETE | what is missing is supplied (KG's base set, a fit, a compare or the re-call); nothing else changes |

No verdict here closes the free GNN (docs/SCREENS.md, section 3b). No model change is adopted on today's pools (the
user's rule of 13 October).

## 7. How it runs

- **In KG's sessions.** zkg_session.py runs each queued fit through its stage's script, each in its own process:
  - `zkg.py fit` for scr-kg- names;
  - `zg6b.py fit` for scr-g6b- names.

  It refuses a name that no stage there fits, or a fit queued twice, before reading any data. KG's take leaves
  scr-g6b- folders to G6b's take. Both changes were made to zkg_session.py and zkg.py in this commit, before any KG
  number (KG, sections 4 and 8).
- **Order.** G6b's fits go first in session 1's two queues, so the longest fits start first:
  - `scr-g6b-zg6b-hp` before zspk's seeds on GPU 1;
  - `scr-g6b-zg6b` before zrckp's seeds on GPU 0.

  A fit not started by the 8-hour start limit moves to the next session. Where a fit runs decides nothing.
- **Time.** zg6's fits took 1,553 s (L-musique) and 2,613 s (L-hotpotqa) on the L40S, train and read together. The
  twin's pass has no edge tensors, so zg6b should cost little more than zg6; g6b_timing.json records it. At three to
  five times slower on the T4, a fit takes roughly 1.5 to 4 hours. It costs nothing (KG, section 8).
- **Commands.** In the kernel, the stub runs the session, and zkg_session runs
  `zg6b.py fit --name N --rows-cap 65536` for each G6b name. On the laptop, after KG's take, compares and verdict:

      python outputs/mp_unified/zg6b.py take --from <the kernel's output folder>
      python outputs/mp_unified/zg6b.py compare-all
      python outputs/mp_unified/zg6b.py verdict

- **What is committed.**
  - Committed: G6b's compares (outputs/screen/scr-g6b-*) and outputs/g6b: the pairs, the re-calls and the verdict,
    which carries the gate statistics and the timing.
  - Never committed: the fits' folders, weights, reads, logs and session files.

## 8. Self-tests, passed before this commit

`python outputs/mp_unified/zg6b.py --selftest` runs on a toy carve and fake files in a temporary folder:
1. **The arm.** zg6b sits on P1c3's link carve, with zg6's class below it. zg6, zg6t and zspk are unchanged.
2. **Where.** train and read refuse anything but a Kaggle T4 on KG's image, and any name that is not G6b's, before
   reading data.
3. **The start.** From seed 0, zg6b draws as zg6 does. Its state is zg6's plus the two gate tensors at 0 (121,806 own
   parameters at hidden 128), and its dropout generator starts where zg6's does.
4. **The two passes.** At the start, s_e and s_t equal zg6's score and its twin's bit for bit: zg6.g6_stack, and zg6
   and zg6t with the same weights. The score is half of each, bit for bit, and base_z is never read.
5. **By hand.** With a random gate, the score equals python loops over the questions and edges: zg6.by_hand for both
   passes, with the shares counted edge by edge.
6. **A question without edges** has s_e = s_t bit for bit, and g = sigmoid(b). A batch without the edges or the KT
   input stops.
7. **The dropout.** There is no draw at read. In training there is one draw from the module's generator:
   - the score equals the by-hand model over the edges zg2.family_keep keeps;
   - the shares count only those edges;
   - the global generator moves as zg6's does.
8. **Gradients** reach every parameter, the gate's included.
9. **Toy fits** through lean_gpu's loop repeat bit for bit. The encoder, the stack, the readout and the gate all move.
10. **The gate beside.** From the toy's link arrays, each question's gate equals the model's own gate and a brute
    count, and g6b_gate.json is gate_stats' output. An array with other bytes stops.
11. **Stamps and take.**
    - It refuses a fit that is not G6b's, and a read on another device.
    - take copies records, reads and the gate file, never models.pt, and leaves KG's fits to KG.
    - take refuses an unread fit, a name that is not G6b's, and a file with other bytes.
12. **One device** across G6b's fit and KG's two bases.
13. **The verdict**, each outcome with its beside rows:
    - ADOPT_FOR_FULL_RUN;
    - NOT_ADOPTED with G6b2 next;
    - NOT_ADOPTED with G6r next, including a GAIN that a raised floor turns WITHIN, and a LOSS alone.

    INCOMPLETE in five cases:
    - without KG's verdict file;
    - with KG's verdict not BASE_READY;
    - without a compare;
    - with the fits on another device than KG's;
    - with a zg6 column that is not KG's zg6 read.
14. Nothing is written under G6b's real folders.

`python outputs/mp_unified/zkg_session.py --selftest` also checks that:
- each name maps to its stage's script, and both scripts exist;
- `run` and `stub` stop on a name that no stage there fits, and on a fit queued twice, before anything else.

`python outputs/mp_unified/zkg.py --selftest` also checks that take leaves a scr-g6b- folder alone.

The text encoder stays frozen. webqsp never trains. Test splits are never read. There is no LLM or transformer, and no
new graph.

## 9. Results

### 27 October 2026: NOT_ADOPTED

zg6b's two fits ran first in KG's session 1, at e0278d1 (section 7), on KG's device. scr-g6b-zg6b ran on GPU 0
(3,134 s) and -hp on GPU 1 (5,151 s), and both exited 0. After KG's BASE_READY, on the laptop at edcff68 (zg6b.py as at
e0278d1): `zg6b.py take`, `compare-all` and `verdict`. Records: outputs/screen/scr-g6b-zg6b-vs-zspk.json and .md and
its -hp twin, and outputs/g6b/g6b-zg6b-pair, g6b-zg6b-pair-recall, g6b-zg6b-vs-zg6-pair, g6b-zg6b-vs-zg6-pair-recall
and screen-verdict (.json and .md).

zg6b against KG's zspk, re-called under KG's zspk null, by G6's rule:
- **3 GAIN:**
  - L-musique metaqa: +0.0114 [+0.0097, +0.0130].
  - L-hotpotqa metaqa: +0.0078 [+0.0061, +0.0094].
  - L-hotpotqa musique: +0.0180 [+0.0119, +0.0243], floor 0.0105.
- **3 LOSS:**
  - L-musique webqsp: -0.0367 [-0.0571, -0.0172].
  - L-hotpotqa hotpotqa: -0.0162 [-0.0201, -0.0124].
  - L-hotpotqa webqsp: -0.0243 [-0.0439, -0.0043]. This one is within 0.00015 of its floor of 0.0242; the call
    stands.
- **Two calls changed** with the re-call. L-musique musique (-0.0094, floor 0.0138) went from LOSS to WITHIN, and
  L-hotpotqa 2wiki (+0.0177, floor 0.0219) from GAIN to WITHIN.

**Next form.** Section 6 fixed it for a GAIN and a LOSS: **G6b2**, a per-row back-off. Its gate also reads the row's own
log1p in-degree per family, over the edges that reach the layers. G6b2 gets its own declared file, on KG's T4 against
KG's bases.

Beside, deciding nothing:
- **Against zg6.** Under zspk's floors, zg6b has 1 GAIN and no LOSS against zg6.
  - The GAIN is L-musique webqsp: 0.2784 against 0.2398, or +0.0386 [+0.0232, +0.0539].
  - The other eleven reads are WITHIN. L-musique musique (-0.0085) was a LOSS before the re-call.
  - On every in-domain read zg6b is within 0.0024 of zg6, so its in-domain GAINs are zg6's.
  - The back-off recovers about half of zg6's loss to zspk on L-musique's webqsp read (-0.0753 becomes -0.0367).
  - It recovers none on L-hotpotqa's zero-shot reads: hotpotqa goes from -0.0159 to -0.0162, webqsp from -0.0210 to
    -0.0243.
- **The gate.** g = sigmoid(a . sh + b) weights the edge pass, and 1 - g weights the twin. a is over the structural,
  NER and kNN shares.

  | split | a | b | structural alone | NER alone | kNN alone | no edges |
  |---|---|---:|---:|---:|---:|---:|
  | L-musique | [0.0025, -0.3908, -0.6181] | -0.3339 | 0.418 | 0.326 | 0.279 | 0.417 |
  | L-hotpotqa | [0.0405, -0.5220, -0.9756] | -0.4484 | 0.399 | 0.275 | 0.194 | 0.390 |

  - Over each dataset's s1eval questions, the gate's mean is 0.34 to 0.40 on L-musique and 0.29 to 0.37 on
    L-hotpotqa.
  - Within a dataset, the 10th to 90th percentiles span 0.008 to 0.053. squad is the exception: 0.30 to 0.40 on
    L-musique and 0.23 to 0.37 on L-hotpotqa.
  - No s1eval question is without edges.
  - So every question leans more on the twin than on the edges, and more so the larger its NER and kNN shares. The
    gate is close to one fixed mix per split, not a per-question switch. G6b2's per-row input is the next form.
- **The MLP share**, (zrckp - rrf) / (zg6b - rrf), on the T4:
  - 0.96 to 0.99 on metaqa, hotpotqa and 2wiki in-domain on L-musique, and on metaqa and 2wiki in-domain on
    L-hotpotqa.
  - Below 0.90 on two in-domain reads of L-hotpotqa: musique at 0.84, and squad at 0.83, where zg6b's gain over rrf is
    only 0.0076.
  - Above 1 where zrckp is above zg6b: L-musique squad, L-musique musique (zero-shot), L-hotpotqa hotpotqa (zero-shot)
    and both webqsp reads.
- **Seconds and memory**, against KG's zg6 of the same split on the same T4:

  | split | train s (zg6) | read s (zg6) | peak GPU GB, train / read (zg6) |
  |---|---|---|---|
  | L-musique | 2,239 (1,942), 1.15x | 875 (773) | 4.89 / 5.24 (4.66 / 5.24) |
  | L-hotpotqa | 4,347 (3,998), 1.09x | 786 (687) | 10.29 / 10.29 (10.29 / 10.29) |

  Peak RAM was about 4.6 GB.
