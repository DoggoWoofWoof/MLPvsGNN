# A7p: A7's scorer input, built and fitted again on P1g-b's pools

Declared 20 October 2026, before any number. The next form A7's verdict named for both arms
(docs/A7_S6_SCORER_INPUT.md, sections 5 and 7), under the user's rule of 13 October that no model change is adopted on
today's pools. Code: `outputs/mp_unified/za7p.py`, which wraps A7's builder (`scripts/a7_s6_cols.py`) and runs A7's
arms (`outputs/mp_unified/za7s6.py`), both unchanged. Numbers here are development numbers.

## 1. Why this, now

**A7** (17 October) gave the lean pair S6's scorer as a fixed input. The scorer (fg-gr) rates each row of a pool from
its node features and from the question's typed chains out of its seeds. Its four numbers a row (f, lm, has and seed)
are columns beside the cache, read by a 97-parameter pointwise head that starts at zero. Both arms were
ADOPT_FOR_FULL_RUN on P1c2's pools, each with one GAIN and no LOSS after the seed-null re-call
(docs/A7_S6_SCORER_INPUT.md, section 7).

Its GAINs, from the re-calls (`outputs/screen/scr-a7-zrckps-pair-recall.json` and `-zspks-`; delta = the arm's R@5
minus its base's, with its 95% CI):

| arm | split | dataset | read | base, arm | delta [95% CI] | floor |
| --- | --- | --- | --- | --- | --- | ---: |
| zrckps | L-musique | webqsp | zero-shot | 0.3627, 0.3848 | +0.0221 [+0.0111, +0.0331] | 0.0102 |
| zspks | L-hotpotqa | webqsp | zero-shot | 0.2680, 0.3016 | +0.0336 [+0.0238, +0.0440] | 0.0233 |

Three more reads came close:
- the MLP's webqsp on L-hotpotqa fell -0.0219 [-0.0337, -0.0097]. It was filed LOSS and re-called WITHIN against its
  0.0233 floor, within 0.0014 of it;
- the GNN's musique zero-shot on L-musique rose +0.0133 [+0.0069, +0.0197]: filed GAIN, re-called WITHIN against its
  0.0720 floor;
- the GNN's musique in domain on L-hotpotqa (Claim 1's read) rose +0.0072 [+0.0015, +0.0125], WITHIN against its
  0.0075 floor.

**The rule of 13 October.** No model change is adopted on today's pools. So A7's verdict named the same next form for
both arms: "the arm fitted again on P1g-b's pools (the scorer and its columns built again on P1g-b's looks), then its
full run" (its section 5), each in its own declared file. This is that refit.

**P1g-b's pools** are P1g2's chosen pools: ALL ≥ 0.95 on every dataset, with about 3.4 times today's rows
(docs/P1GC_ADOPTED_ON_P1GB.md, section 1). P1g-b refit the lean pair on them, unchanged, and was NOT_ADOPTED
(`outputs/p1gb/screen-verdict.json`, 2026-10-17T17:41:13Z). P1g-c (declared 18 October) puts them under the adopted
models, and its verdict decides the pools.

**The pools alone moved the same reads.** On these five reads, P1g-b's refit moved the lean pair
(`outputs/screen/scr-p1gb-zrckp-pair-recall.json` and `-zspk-`; delta = the model on P1g-b's pools minus the same model
on P1c2's):

| read | A7's columns, on P1c2's pools | the pools alone, on the base arm | the base on P1g-b's pools (A7p's base R@5) |
| --- | --- | --- | ---: |
| zrckps, L-musique webqsp (A7's GAIN) | +0.0221, GAIN | zrckp +0.0141 [+0.0015, +0.0257], GAIN | 0.3768 |
| zrckps, L-hotpotqa webqsp | -0.0219, WITHIN (filed LOSS) | zrckp -0.0111 [-0.0245, +0.0025], WITHIN | 0.3364 |
| zspks, L-hotpotqa webqsp (A7's GAIN) | +0.0336, GAIN | zspk +0.0129 [-0.0003, +0.0274], WITHIN | 0.2808 |
| zspks, L-musique musique, zero-shot | +0.0133, WITHIN (filed GAIN) | zspk +0.0157 [+0.0098, +0.0221], WITHIN (filed GAIN) | 0.5169 |
| zspks, L-hotpotqa musique, in domain | +0.0072, WITHIN | zspk -0.0092 [-0.0145, -0.0039], LOSS | 0.5662 |

So on the reads that decided A7, A7p's bases already differ from A7's. On webqsp the pools alone moved the lean pair
+0.0141 and -0.0111 (MLP), and +0.0125 and +0.0129 (GNN). On musique they moved the GNN zero-shot about as far as A7's
columns did. In domain they cost zspk its one LOSS, on the read the columns lifted. Only a refit on P1g-b's pools says
whether the columns still help there.

**The pools also change the columns.** The scorer reads each row's place in its pool: its first-stage rank, degree,
walk masses and best neighbour's question cosine, and the chains out of the question's seeds over the pool (A7's
section 2). A7's builds hold each question's whole pool, as its look holds it. P1g-b's pools hold about 3.4 times the
rows, so every one of these inputs changes. A7's scorer was also fitted and standardised on P1c2's looks. So A7p builds
the inputs again on P1g-b's looks, fits the scorer again and computes the columns again. It keeps A7's code and
settings, and nothing else. A7's four scorer fits kept epochs 12 and 9 (L-musique), 12 and 10 (L-hotpotqa), with
γ 0.91 and 0.92 (L-musique), 0.86 and 0.85 (L-hotpotqa). A7p's verdict reports its own fits beside them.

## 2. The form

A7's arms, unchanged: `za7s6.ZRCKPS` and `za7s6.ZSPKS` on A7's carves, with A7's head and settings
(docs/A7_S6_SCORER_INPUT.md, sections 2 to 4). The scorer, its builds and its columns are A7's, made again on P1g-b's
looks and caches. Only the pools change.

- **The builder, wrapped.** za7p.py runs A7's `sbuild`, `sfit` and `cols` unchanged, with A7's settings: the same
  graphs, hops, levels, transforms, regimes, passage typing, folds, optimiser and kept-epoch rule. It changes only
  where they read and write:
  - **reads:** P1g-b's looks (`outputs/p1gb/look`), under P1g-c's look alias, and P1g-b's caches
    (`outputs/p1gb/cache`). The columns are four per cache row of P1g-b's cache;
  - **writes:** the builds to `outputs/a7p/cache`, the scorer fits to `outputs/a7p/fits` and the columns to
    `outputs/a7p/cols`. A7's, P1g-b's, P1g-c's and P1c2's roots are refused;
  - **records:** A7's own writers, with A7p's declaration, and A7's settings but for one field (the pools: P1g-b's
    looks). The stage names stay the builder's (A7-build, A7-fit, A7). Each record's placement names A7p, the look and
    alias roots, and the sha256 of each alias read.
- **The look alias.** P1g-c rebuilt P1g-b's looks. Each rebuilt record holds its run's seconds, time and peak memory.
  So its sha256 is not the one P1g-b's cache parts stamped, and A7's builds stop ("built from another look";
  docs/P1GC_LOOK_ALIAS.md, section 1). That file's section 4 fixed this case: "A7p (A7's next form) reads P1g-b's looks
  beside its caches. It builds under the same alias, in the same carves." So every builder call runs inside
  `zp1gcla.on_alias`, the same swap of `lean_cache.look_chunks` that `zp1gcla.py run` makes, in one process:
  - before the call, each carve's alias must be P1g-c's ALIAS of that carve, with its rebuilt sha256s equal to the
    records on disk;
  - after the call, the builder must have read each carve under its alias.

  A missing, REFUSED or outdated alias stops the call, and nothing is built.
- **One thread.** A7's builds and fits ran at one thread. When `sbuild`, `sfit` or `cols` is the command, za7p.py sets
  the three thread variables to 1 before numpy loads. It refuses to build otherwise.
- **The arms' roots.** The arms read P1g-b's caches, KT columns, links and chains inside `zp1gb.on_p1gb()`, as A5cp's
  arms do: the same roots P1g-c's own context sets. A7's carve classes read A7p's columns, each split's own
  (`outputs/a7p/cols/<split>/…`). All are restored after.
- **The gate.** Every fit and read checks P1g-c's gate first (`zp1gc.gate_open`), as A5cp's do. Each refuses while the
  gate is shut, missing or another stage's, and when a P1g-b record changed after the gate opened. A fit also checks
  P1g-b's own gate (`zp1gb.gate_passed`) and that the KT records it reads are P1g-b's. Every column record it reads
  must be A7p's, of its split. A fit stamps the column records of its training carves and of the six s1eval carves,
  and a read refuses s1eval columns that changed after its fit.
- **The MLP rule.** A7's columns comply, as A7's section 2 says, and A7p changes nothing in them:
  - **no message:** the columns are fixed numbers for each row, computed in the builds before either model scores a
    row. zrckps has no propagation step, and its carve holds no edge (the self-test checks both). The scorer's graph
    inputs (degree, walk masses, the best neighbour's question cosine, the chain masses m_c(v)) are fixed features of
    the pool, computed with it, as zrc's chains and the KT walk positions are. Nothing learned passes along an edge;
  - **no other row's score:** no column reads the MLP's, the GNN's or a base's score. f and lm are the scorer's values
    for the row itself;
  - **nothing normalised over the rows from a score:** p0 is a softmax over the question's chains, from chain
    features, not over its rows. m_c(v) is a walk's share of the pool, from the graph and the chain's relation types,
    not from a score. fg-gr was chosen over em-gr because em-gr's rounds reweight chains by a softmax of the score over
    the question's nodes (A7's section 1). The first-stage rank in f is allowed: "Ranks or z-scores of first-stage
    retrieval values within the pool are computed before any row is scored, as zret's z-scores are."
    (docs/VERDICT_AUDIT_2026_10_17.md, section 2);
  - zspks passes zsp's one message, as zspk does. Its head is added before zsp's propagation reads the score.
- **The method rules.** The encoder is frozen: the builds read the look's own embeddings. Passages are pseudo-typed
  as A7's are, by 256 clusters (spherical k-means) of the endpoint projections of the carve's structural triples. No
  LLM, transformer or sentence-level embedding enters. The scorer is fitted on gold labels (chainscore19's node and
  chain losses), never on a GNN's scores. webqsp never trains: it is in no split's training datasets, and a split's
  scorer never trains on its held-out dataset. The carves read are step 1's fit and s1eval carves. The s1eval carves
  are dev, validation or train_holdout questions (outputs/step1/carves.json), so no test split is read.
- **Serving cost.** A7's: the columns need the pool's chains and the scorer's two MLPs at query time, and the timed
  form is declared with the full run if one follows (A7's section 4). What the bigger pools cost is P1g-c's question,
  not this file's.

## 3. The arms

| arm | model | carve | its base, on P1g-b's pools |
| --- | --- | --- | --- |
| zrckps | A7's MLP: zrckp's model plus the head at the top (MRO: ZRCKPS, ZRCKP, KTMixP, ZRM, ChainMatch, ZRet) | A7's `za7s6.A7KTChainCarveZRC`: zrckp's carve (`zp1c3.KTChainCarveZRC`) with the columns beside | P1g-b's `scr-p1gb-zrckp`, `-hp` |
| zspks | A7's GNN: zspk's model with the head added to the KT head before zsp's propagation reads the score (MRO: ZSPKS, ZProp, A7Mix, KTMix, ZRM, ChainMatch, ZRet) | A7's `za7s6.A7KTLinkCarveBase`: zspk's carve (`zp1c3.KTLinkCarveBase`) with the columns beside | P1g-b's `scr-p1gb-zspk`, `-hp` |

Each split's fits read that split's columns only. A fit of L-musique never sees a scorer that trained on musique, nor
one of L-hotpotqa on hotpotqa.

## 4. The builds, the fits and the call

A7's, on P1g-b's pools:
- **Builds:** A7's nine stored builds, of the training datasets' fit carves: metaqa's fit carve five times (id and the
  four transforms), squad, hotpotqa and 2wiki fit, and musique s1fit. The six s1eval carves are built inside their
  column items, in `outputs/a7p/cache/tmp`, and removed there.
- **Scorer fits:** four, one per split and fold. Each trains on its split's four training datasets' builds (metaqa's
  five), never on its held-out dataset, with A7's optimiser and kept-epoch rule.
- **Columns:** eleven items: the five fit carves (on their stored id builds) and the six s1eval carves, for each split
  that reads the carve. Each question's columns come from the scorer of the other fold.
- **Card fits:** one per arm on L-musique and one on L-hotpotqa (`scr-a7p-<arm>`, `-hp`), four in all. They run on
  P1g-b's roots, with P1g-b's KT columns and A7p's columns, at the card fits' settings (variant p, config
  2e-3:1e-4:0.1:8:2, seed 0, hidden 128).
- **Reads:** each fit on the six s1eval carves of P1g-b's pools at p@swa.
- **Compares:** each fit against P1g-b's card fit of its base arm on the same split (zrckps: scr-p1gb-zrckp and -hp;
  zspks: scr-p1gb-zspk and -hp), with step 1's fit beside. Both fits read the same pools, questions and gold totals.
- **The re-call's base screens:** A5cp's (`outputs/a5cp/base-<base>-<split>.json`, items `a5cp-base-*`): P1g-b's four
  card fits against step 1's. A7p's bases are the same four fits, so it reuses them and makes none.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), per arm. GAIN,
  LOSS and WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

A7's rule, per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A7p's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (docs/SCREENS.md, section 3b), and it names each arm's next form (section 5).

**Reported beside, deciding nothing:**
- webqsp zero-shot (A7's GAINs; webqsp never trains) and metaqa in domain (the typed graph whose transforms the
  scorer trains on), on both splits and arms;
- musique in domain (L-hotpotqa, Claim 1's read), both arms side by side;
- hit@1 and FC@5 per read, from the compares;
- each read's delta beside A7's delta and call on the same read on P1c2's pools (from A7's re-calls);
- per scorer fit: its kept epoch, its held-out objective (in all and per dataset), γ and ε, its time and its peak
  memory, beside A7's (its section 7);
- the disk: the bytes under A7p's builds, columns and scorer fits; each stored build's size, rows, time and peak
  memory; and the folders' lifetime (section 6).

## 5. What each outcome leads to (fixed now)

Per arm:

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | A7's columns join the full run's models. That full run is one file, declared before its numbers, after P1g-c's, A5cp's, A7p's and A3vp's verdicts, with the composition P1g-c's section 5 sets. Each of its splits needs its own scorer fits and columns, built as A7p's are |
| NOT_ADOPTED, with a GAIN and a LOSS | **A7k** on P1g-b's pools: the four columns split by the graph's kind (the KB graphs' and the pseudo-typed passage graphs'; eight columns, the other kind's four at 0), the back-off form A7's section 5 names, on both models, its own declared file |
| NOT_ADOPTED, with no GAIN | **A7e** on P1g-b's pools, A7's form for this outcome: on zspk, chainscore20's carried scorer ena-gr in the scorer's place; on zrckp, the fg-gr scorer fitted on metaqa's five builds alone. Its own declared file. The idea stays open |
| INCOMPLETE | the missing re-call is run; nothing else changes |

**Why these.**
- **ADOPT.** The columns then passed on both pools. They join the full run as the other adopted changes do, in the
  composition P1g-c's section 5 sets (its pools, its MLP and its GNN). If both arms pass, the columns join both models,
  and F1's rule (one input set for both) holds. If only one passes, the full run's file names the columns as a
  difference between its two models. The verdict records which.
- **The full run's scorers.** Each of the full run's six splits needs its own scorer fits and columns, since a split's
  scorer never trains on its held-out dataset. L-metaqa needs one more decision. A7's builder standardises the
  scorer's inputs by metaqa's id build, and it refuses a split whose training datasets do not start with metaqa. So
  the full run's file declares L-metaqa's scorer before its numbers.
- **A GAIN with a LOSS.** A7's own form for this outcome (its section 5): the columns split by the graph's kind, so
  the KB graphs' chain weights cannot move the passage graphs' head weights.
- **No GAIN.** The columns gained on P1c2's pools and not on P1g-b's. Section 1 shows the pools moved the same reads:
  webqsp rose on three of the four lean fits, one of them past the null, and the GNN's musique in domain fell. So
  either the bigger pools already give what the columns gave, or the scorer refit on them carries less. That closes
  this form only. The idea had a signal, so it stays open (docs/SCREENS.md, 3b). A7's own form for no GAIN is A7e: a
  scorer with the EM rounds for the GNN, and the scorer of S6's filed setting for the MLP.
- **INCOMPLETE.** As A7's.

**The pools.** Which pools the full run uses is P1g-c's call (its section 5). An arm's columns join the full run if it
passed its screen on the pools that run uses:
- **on P1g-b's pools** (P1g-c's pools call ADOPT_FOR_FULL_RUN), this file's verdict decides, per arm;
- **on P1c2's pools** (P1g-c's pools call NOT_ADOPTED), A7's own verdict on those pools decides (both arms
  ADOPT_FOR_FULL_RUN, docs/A7_S6_SCORER_INPUT.md, section 7), and this file's verdict is reported beside, deciding
  nothing.

## 6. How it runs

**Self-test.** `python outputs/mp_unified/za7p.py --selftest` passes on the laptop before this commit (16.8 and 24.1 s
over two runs, 22 to 29 s with its imports; run from the repo root, as on the host: the cache-root check resolves
against it). On toy carves with the KT input, rmatch's chains and zlink's links, A7's toy columns, and a toy metaqa
look rebuilt after its cache parts stamped it, it checks:
- **the arms:** A7's own objects, on A7's carves. Each model is its base arm's (P1g-b's zrckp or zspk) plus A7's head.
  The MLP's carve holds no edge, and its model has no message step. webqsp is in no training split. The bases are
  P1g-b's card fits. A7p's settings are A7's but for the pools, and the builder's constants are A7's. The thread
  variables are set before numpy loads, and a build refuses without them;
- **the roots:** inside, P1g-b's (caches, chains, links, KT columns, gate and declaration): the same roots P1g-c's arms
  read. With A7p's redirect, A7's carves read A7p's columns, and A7's and the builder's records are A7p's. A7's,
  P1g-b's, P1g-c's and P1c2's roots are refused for the columns. All are restored after;
- **A7's model, bit for bit:**
  - A7's carve classes, pointed at A7p's columns, give the same batches as on A7's (the same arrays under A7p's
    records), and each refuses the other's records;
  - from the same seed, each arm has the same state and the same draws as A7's, and with the head at zero it scores as
    its base arm;
  - with the head and the KT head set, the same scores and every gradient, in eval and in training;
  - toy fits through lean_gpu's loop on A7p's columns inside P1g-b's roots equal A7's outside them, bit for bit (every
    state and the SWA state), and move the head;
- **A7's builder, under the alias,** on a toy metaqa look whose records were written again after its cache parts
  stamped them, as P1g-c's rebuild left P1g-b's:
  - each wrapper refuses a foreign root, a carve A7p does not store, a missing, REFUSED or outdated alias, and a call
    that never read the alias. Nothing is built;
  - the commands pass the builder every root of its signature (A7p's by default, and P1g-b's looks) and A7p's
    placement;
  - the real build, two scorer fits and the columns of the fit and s1eval carves run under the alias and write A7p's
    records. The s1eval build is made and removed inside its column item;
  - without the alias, A7's own check stops ("built from another look"). Outside A7p's redirect, A7's loaders refuse
    A7p's records;
- **train and read:**
  - before any gate or data, they refuse the lean pair's and P1g-c's arms (zrckp, zspk, zrckpr, zspx, zspxr), another
    split, another fit name (A7's included), a foreign columns root, P1c2's caches or none, and the CPU. Read refuses
    A7's, P1g-b's, another arm's and the wrong split's fits;
  - on toy records of P1g-b's, train and read refuse while P1g-c's gate is shut, after a record changed, and when it is
    another stage's or missing. With it open, a fit refuses a missing or failed P1g-b gate, KT records that are not
    P1g-b's, and columns that are missing or not A7p's of its split (A7's, built on P1c2's looks, or another split's);
  - with all in place, the fit runs on P1g-b's roots with A7's carves on A7p's columns. It is stamped A7p's: both
    gates, the snapshot, its base fit, the settings, and the column and KT records it read. The read refuses s1eval
    columns changed after the fit;
- **the calls:** relz's pair and re-call per arm against P1g-b's fits of its base arm, stamped A7p's, with A5cp's base
  screens. A screen decided against A7's base (P1c2's pools), and base screens of A7's bases, are refused. relz is
  restored after;
- **the verdict:** ADOPT, the two NOT_ADOPTED outcomes, four INCOMPLETE cases (another stage's re-call, a missing
  one, eleven reads, another arm's), the next forms, what is beside (FC@5, hit@1, A7's delta and call, the scorer
  fits, the disk and the lifetime), and the F1 note. A7's two re-calls, read beside, hold twelve reads each;
- **nothing written** under A7's, A7p's, P1g-b's, P1g-c's or A5cp's roots, nor A7's, A7p's or P1g-b's screens. The
  thread variables and torch's threads are as they were.

**Host items** (41, `a7p-*`):
- **Builds:** 9 (`a7p-build-<dataset>-<carve>-<transform>`), 1 CPU each. Each follows its carve's alias
  (`p1gc-la-<dataset>-<carve>`). Memory is A7's measured peak for the same build, × the dataset's rows factor + 1 GB.
  The factors are P1g-b's (docs/P1GB_POOLS_REFIT.md, section 6): metaqa ×1.68, squad ×1.00, musique ×3.15,
  hotpotqa ×11.60, 2wiki ×14.14, webqsp ×2.24.
- **Scorer fits:** 4 (`a7p-fit-<split>-f<fold>`), 1 CPU each, each after its split's eight builds. Memory is A7's
  measured peak for the same fit, × 5.1 (L-musique) or 3.8 (L-hotpotqa) + 2 GB. These are the factors of the split's
  datasets, weighted by their fit questions (metaqa's counted 3.5 times, for its five builds). Weighted by A7's probe
  bytes (its section 6) they would be about 2.3 to 2.4, so this errs high. The fits are short.
- **Columns:** 11 (`a7p-cols-<dataset>-<carve>`), 1 CPU each. Memory is A7's measured peak for the same item, × the
  dataset's factor + 1 GB. The six s1eval items each follow their carve's alias and the four scorer fits. The five
  fit-carve items each follow their stored id build and the fits that read the carve: all four for metaqa, squad and
  2wiki, L-musique's two for hotpotqa, and L-hotpotqa's two for musique.
- **The screen:** 17 items. Per arm: two trains, two reads, two compares, a pair and a re-call. Then the verdict.
  - **Trains and reads** run on the card at `--frac 0.92`, share 0.94 and 1.1 CPUs, one at a time, as P1g-b's,
    P1g-c's and A5cp's do. The arms read what the base arms read, plus the columns. So memory is P1g-b's measured peak
    for the base arm's item of the same split, × 1.1 + 2 GB. Trains follow P1g-c's gate (`p1gc-gate`), their split's
    four fit-carve column items, and the six s1eval column items, which the fit stamps. Each read follows its train,
    the gate and the six s1eval column items.
  - **Compares** (1 CPU, 4 GB) follow their read. **Pairs** (1 CPU, 2 GB) follow both of the arm's compares.
    **Re-calls** (1 CPU, 2 GB) follow the pair and A5cp's two base screens of the arm's base (`a5cp-base-<base>`,
    `-hp`). **The verdict** (1 CPU, 2 GB) follows both re-calls.
- **Inputs:** P1g-c's train list (45 files), the ten more that A7's build items list (A7's builder, the six modules it
  loads, chainpop17, chainscore18 to 21 and reltype11, then za7s6.py and two configs), zp1gcla.py and za7p.py: 57
  files, which hold za7p.py's import closure with A7's builder loaded. The re-calls add the four null files, and the
  verdict adds A7's two re-calls.

**The releases.** A7p's 24 look readers (its builds, scorer fits and column items) join the dependencies of P1g-c's
`p1gc-release-look-<dataset>` items. docs/P1GC_LOOK_ALIAS.md (section 6) fixed this: "Once declared, they are added
to the release items' dependencies." Per dataset:

| release item | A7p's items it waits for |
| --- | --- |
| `p1gc-release-look-metaqa` | 11: its five builds, the four scorer fits, its two column items |
| `p1gc-release-look-squad` | 7: its build, the four scorer fits, its two column items |
| `p1gc-release-look-2wiki` | 7: its build, the four scorer fits, its two column items |
| `p1gc-release-look-hotpotqa` | 5: its build, L-musique's two scorer fits, its two column items |
| `p1gc-release-look-musique` | 5: its s1fit build, L-hotpotqa's two scorer fits, its two column items |
| `p1gc-release-look-webqsp` | 1: its s1eval column item |

So the clean-up keeps each of P1g-b's looks until A7p's last reader of it ends. If a reader fails, the release is
dropped behind it. The clean-up then keeps the folders of the failed item and of the items dropped behind it for 48 h
after its launch (docs/P1GC_LOOK_ALIAS.md, section 6).

**Disk** (estimates; the verdict reports the measured sizes):
- **stored builds:** about 9 GB at A7's probe rates (its section 6: bytes a question, at regime 1, on P1c2's pools),
  times each fit carve's questions (outputs/step1/carves.json) and its dataset's factor. That is metaqa about 4.0 GB
  (its five builds hold its fit carve 3.5 times), musique 2.4, 2wiki 1.4, hotpotqa 1.0 and squad 0.2. A fit build also
  holds the pair features of its other regimes (six in all; three for al4, sp4 and rf), which the probes did not. So
  the stored builds may take up to about 30 GB;
- **s1eval builds:** made in `outputs/a7p/cache/tmp` inside their column items, at regime 1, and removed there. About
  9 GB if all six run at once: metaqa 1.9, squad 0.3, musique 1.3, hotpotqa 1.2, 2wiki 2.9, webqsp 1.6;
- **columns:** four float16 numbers (8 bytes) a cache row, for each split that reads the carve. **Scorer fits:** small;
- **lifetime:** `outputs/a7p` stays while an A7p item is queued or running (za7p.py names it). After that, the builds
  and columns are build folders the clean-up may remove, and the scorer fits stay. While A7p's items are queued, the
  clean-up also keeps the roots A7's builder names (A7's, P1c2's caches and looks, M3B's relations);
- **the floor:** A7p's builds, scorer fits and column items write gigabytes. Below its 30 GB disk floor the feeder
  sends only items that write little, but since 20 October a writer that a pending release waits on still goes
  there, one at a time, and below the 15 GB hard floor only releases go (hostfeed2's release path, c61361c). Each
  release of P1g-b's looks now waits for A7p's readers of them, so those readers are such writers. In the feeder's
  list A7p's items go after A5cp's and A3vp's, before P1g-c's own items on the rebuilt looks (its aliases,
  residuals and releases), so each build is tried as soon as its alias is made;
- **the room:** beside P1g-c's link, chain and residual builds (about 32 GB, docs/P1GC_LOOK_ALIAS.md, section 6),
  the drive's 55 GB was too little for A7p's builds as well before the releases free P1g-b's 54 GB of looks. So on
  20 October, before this file, the clean-up was given nine caches of the pools before P1c2 (29.95 GB, regen files
  only; `outputs/host_ops/release/cold-pre-p1c2-caches.json`): Step 1's per-dataset features cache and the zlink,
  zrc and rmatch default caches. No queued item names them, and none of their arrays was read in the 24 hours
  before.

**ETA** (estimates; the verdict reports each build's time):
- **the CPU items:** A7's whole chain was a few CPU-hours on P1c2's pools: "musique's s1fit build and webqsp's s1eval
  build are the longest (about 45 and 40 minutes)" (A7's section 6). If time grows with the rows, those take about 2.4
  and 1.5 hours here (× 3.15 and × 2.24). At A7's probe rates, 2wiki's s1eval column item takes about 4.1 hours and
  its fit build about 2. The builds run side by side, one CPU each; the scorer fits follow the builds, and the column
  items the fits. So the columns are ready about 7 hours after the aliases. At A7's 40-minute bound for its other
  builds, 2wiki's build and its s1eval item take up to about 9.4 hours each (× 14.14), or 19 hours in all. Time may
  grow faster than the rows: a look's cost was about quadratic in the pool (docs/STEP4E_PAPER_POOLS.md);
- **the card:** P1g-b's fits took 972 to 1,739 s each, one at a time with the full card (share 0.94), and P1g-c puts a
  read at about 7 minutes (docs/P1GC_ADOPTED_ON_P1GB.md, section 6). So A7p's four fits and reads take about 1.6 to
  2.4 hours of the card. They can start once P1g-c's gate opens and A7p's columns are made. They share the card one at
  a time with P1g-c's six fits and reads (about 3.5 hours, the same section), A5cp's four and A3vp's two, in the order
  the queue sets. The CPU screen items take minutes.

**The commands.** One arm and split are shown, with one build, scorer fit and column item of each kind. zspks and
L-hotpotqa (names ending `-hp`) follow the same pattern. On the host the builds, scorer fits and column items run under
`retry_cmd.py`, and the trains and reads under `retry_cmd.py --cuda` and `cuda_alloc.py --frac`, as in the items.

    python outputs/mp_unified/za7p.py sbuild metaqa fit --transform al4 --host
    python outputs/mp_unified/za7p.py sfit L-musique 0 --host
    python outputs/mp_unified/za7p.py cols metaqa fit --host
    python outputs/mp_unified/za7p.py cols webqsp s1eval --host
    python outputs/mp_unified/za7p.py train --split L-musique --name scr-a7p-zrckps --arm zrckps --device cuda \
        --host --cache-root outputs/p1gb/cache
    python outputs/mp_unified/za7p.py read --name scr-a7p-zrckps --device cuda --host --cache-root outputs/p1gb/cache
    python outputs/mp_unified/za7p.py compare --new outputs/screen/fits/scr-a7p-zrckps \
        --base outputs/screen/fits/scr-p1gb-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a7p-zrckps
    python outputs/mp_unified/za7p.py pair --arm zrckps \
        --screens outputs/screen/scr-a7p-zrckps.json,outputs/screen/scr-a7p-zrckps-hp.json \
        --out outputs/screen/scr-a7p-zrckps-pair
    python outputs/mp_unified/za7p.py recall --arm zrckps \
        --null outputs/screen/scr-null-s1.json,outputs/screen/scr-null-s2.json,outputs/screen/scr-null-s1-hp.json,outputs/screen/scr-null-s2-hp.json \
        --pair outputs/screen/scr-a7p-zrckps-pair.json --out outputs/screen/scr-a7p-zrckps-pair-recall
    python outputs/mp_unified/za7p.py verdict

## 7. Results

Not run yet.
