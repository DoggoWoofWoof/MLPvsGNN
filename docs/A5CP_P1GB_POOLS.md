# A5cp: A5c's coupled kind offsets, fitted again on P1g-b's pools

Declared 19 October 2026, before any number. The next form A5c's verdict named for both arms
(docs/A5C_KIND_COUPLED.md, sections 5 and 7; `outputs/a5c/screen-verdict.json`), under the user's rule of 13 October
that no model change is adopted on today's pools. Code: `outputs/mp_unified/za5cp.py`. Numbers here are development
numbers.

## 1. Why this, now

**A5c** (18 October) gave the lean pair zck's kind offsets in their coupled form. On a batch from a graph with typed
relations (metaqa and webqsp), zret's output layer and rrf's base weight move by learned offsets, trained jointly with
the shared weights. Both arms were ADOPT_FOR_FULL_RUN on P1c2's pools, each with two GAINs and no LOSS after the
seed-null re-call (`outputs/a5c/screen-verdict.json`, 2026-10-18T14:59:12Z).

Its GAINs, from the re-calls (`outputs/screen/scr-a5c-zrckpc-pair-recall.json` and `-zspkc-`; delta = the arm's R@5
minus its base's, with its 95% CI):

| arm | split | dataset | read | base, arm | delta [95% CI] | floor |
| --- | --- | --- | --- | --- | --- | ---: |
| zrckpc | L-musique | webqsp | zero-shot | 0.3627, 0.3816 | +0.0189 [+0.0089, +0.0294] | 0.0102 |
| zrckpc | L-hotpotqa | webqsp | zero-shot | 0.3475, 0.3714 | +0.0239 [+0.0122, +0.0359] | 0.0233 |
| zspkc | L-hotpotqa | musique | in-domain | 0.5754, 0.5837 | +0.0083 [+0.0036, +0.0131] | 0.0075 |
| zspkc | L-hotpotqa | webqsp | zero-shot | 0.2680, 0.3200 | +0.0520 [+0.0392, +0.0648] | 0.0233 |

musique zero-shot rose too on L-musique (MLP +0.0084, GNN +0.0224): filed GAIN, re-called WITHIN against its 0.0720
floor.

**The rule of 13 October.** No model change is adopted on today's pools. So A5c's verdict named the same next form for
both arms: the arm fitted again on P1g-b's pools, then its full run, each in its own declared file. This is that refit.

**P1g-b's pools** are P1g2's chosen pools: ALL ≥ 0.95 on every dataset, with about 3.4 times today's rows
(docs/P1GC_ADOPTED_ON_P1GB.md, section 1). P1g-b refit the lean pair on them, unchanged, and was NOT_ADOPTED
(`outputs/p1gb/screen-verdict.json`, 2026-10-17T17:41:13Z). P1g-c (declared 18 October) puts them under the adopted
models, and its verdict decides the pools.

**The pools alone moved the same reads.** On the four reads where A5c gained, P1g-b's refit moved the lean pair
(`outputs/screen/scr-p1gb-zrckp-pair-recall.json` and `-zspk-`; delta = the model on P1g-b's pools minus the same model
on P1c2's):

| read | A5c's offsets, on P1c2's pools | the pools alone, on the base arm | the base on P1g-b's pools (A5cp's base R@5) |
| --- | --- | --- | ---: |
| zrckpc, L-musique webqsp | +0.0189, GAIN | zrckp +0.0141 [+0.0015, +0.0257], GAIN | 0.3768 |
| zrckpc, L-hotpotqa webqsp | +0.0239, GAIN | zrckp -0.0111 [-0.0245, +0.0025], WITHIN | 0.3364 |
| zspkc, L-hotpotqa musique | +0.0083, GAIN | zspk -0.0092 [-0.0145, -0.0039], LOSS | 0.5662 |
| zspkc, L-hotpotqa webqsp | +0.0520, GAIN | zspk +0.0129 [-0.0003, +0.0274], WITHIN | 0.2808 |

So A5cp's bases differ from A5c's on exactly these reads. On webqsp the pools alone moved the lean pair +0.0141 and
-0.0111 (MLP), and +0.0125 and +0.0129 (GNN). The GNN's musique in domain, which the offsets lifted, is the read where
the pools cost zspk its one LOSS. Only a refit on P1g-b's pools says whether the offsets still help there.

## 2. The form

A5c's arms, unchanged: `za5c.ZRCKPC` and `za5c.ZSPKC` on their own carves, with A5c's offsets and settings
(docs/A5C_KIND_COUPLED.md, section 2). Only the pools change.

- **The offsets.** On a typed batch, zret's score is (base_w + kd_base) · z(rrf) + (W + kd_w) · h + (b + kd_b). Any
  other batch is zret's forward. The three offsets start at zero and train jointly with the shared weights. Both splits
  train on metaqa, so both fits move them. webqsp never trains.
- **The roots.** The arms read P1g-b's caches, KT columns, links and chains, as P1g-c's arms do. They run inside
  `zp1gb.on_p1gb()`, which sets the same roots that P1g-c's own context sets for every carve A5c's arms read. The MLP's
  carve reads zrc's chains. The GNN's carve reads rmatch's chains and zlink's links.
- **No build.** A5c's carves are the lean pair's own (`zp1c3.KTChainCarveZRC` and `zp1c3.KTLinkCarveBase`): the ones
  P1g-b's fits read. A typed batch is one that carries chain entries. The chain carves add those on metaqa and webqsp,
  from zrc's chains (the MLP) and rmatch's chains (the GNN). Those chains, the links and the KT columns are the records
  P1g-c rebuilds and its gate checks (`zp1gc.py`, RECS: links, rmatch, zrc and kt). So A5cp needs nothing beyond what
  zrckp and zspk read, and it builds nothing.
- **The gate.** Every fit and read checks P1g-c's gate first (`zp1gc.gate_open`), as zp1gc.py does. Each refuses while
  the gate is shut, missing or another stage's, and when a P1g-b record changed after the gate opened. A fit also checks
  P1g-b's own gate (`zp1gb.gate_passed`). Fits and reads check that the KT records they read are P1g-b's.
- **The MLP rule.** zrckpc is A5c's MLP as filed. The offsets are weights. It passes no message, reads no other row's
  score, and adds nothing normalised over the question's rows. zspkc passes zsp's one message, as zspk does.
- **Serving cost.** A5c's: one added product per row on a typed graph. The passage graphs serve as the base does. What
  the bigger pools cost is P1g-c's question, not this file's.

## 3. The arms

| arm | model | carve | its base, on P1g-b's pools |
| --- | --- | --- | --- |
| zrckpc | A5c's MLP: zrckp's model with A5CRet just above zret (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, A5CRet, ZRet) | zrckp's (`zp1c3.KTChainCarveZRC`) | P1g-b's `scr-p1gb-zrckp`, `-hp` |
| zspkc | A5c's GNN: zspk's model with A5CRet just above zret (MRO: ZSPK, ZProp, KTMix, ZRM, ChainMatch, A5CRet, ZRet) | zspk's (`zp1c3.KTLinkCarveBase`) | P1g-b's `scr-p1gb-zspk`, `-hp` |

## 4. The fits and the call

A5c's, on P1g-b's pools:
- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-a5cp-<arm>`, `-hp`), four in all. They run on
  P1g-b's roots with P1g-b's KT columns, at the card fits' settings (variant p, config 2e-3:1e-4:0.1:8:2, seed 0,
  hidden 128).
- **Reads:** each fit on the six s1eval carves of P1g-b's pools at p@swa.
- **Compares:** each fit against P1g-b's card fit of its base arm on the same split (zrckpc: scr-p1gb-zrckp and -hp;
  zspkc: scr-p1gb-zspk and -hp), with step 1's fit beside. Both fits read the same pools, questions and gold totals.
- **The re-call's base screens:** P1g-b's four fits against step 1's (`outputs/a5cp/base-<base>-<split>.json`). They
  are this stage's own. P1g-b's base screens (`outputs/p1gb/base-*.json`) are of its bases on P1c2's pools. P1g-b's
  compares put those bases first, with step 1's fit beside, and the re-call needs step 1's first.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), per arm. GAIN,
  LOSS and WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

A5c's rule, per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A5cp's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (docs/SCREENS.md, section 3b), and it names each arm's next form (section 5).

**Reported beside, deciding nothing:**
- webqsp zero-shot (A5c's GAINs; webqsp never trains) and metaqa in domain (the typed graph the offsets train on), on
  both splits;
- musique in domain (L-hotpotqa, Claim 1's read), both arms side by side;
- hit@1 and FC@5 per read, from the compares;
- each read's delta beside A5c's delta and call on the same read on P1c2's pools (from A5c's re-calls);
- per fit, the typed and untyped training forwards it stamped.

## 5. What each outcome leads to (fixed now)

Per arm:

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | A5c's coupled offsets join the full run's models. That full run is one file, declared before its numbers, after P1g-c's, A5cp's and A7p's verdicts, with the composition P1g-c's section 5 sets |
| NOT_ADOPTED, with a GAIN and a LOSS | **A5cb** on P1g-b's pools: the coupled offsets bounded (each offset 0.2 tanh of its parameter), its own declared file |
| NOT_ADOPTED, with no GAIN | **A5i** on P1g-b's pools: the graph's kind as a 0/1 input through R1's pointwise head (the learned back-off), on both models, its own declared file. The idea stays open |
| INCOMPLETE | the missing re-call is run; nothing else changes |

**Why these.**
- **ADOPT.** The offsets then passed on both pools. They join the full run as the other adopted changes do, in the
  composition P1g-c's section 5 sets (its pools, its MLP and its GNN). If both arms pass, the offsets join both models,
  and F1's rule (one input set for both) holds. If only one passes, the full run's file names the offsets as a
  difference between its two models. The verdict records which.
- **A GAIN with a LOSS.** A5c's own form for this outcome (its section 5): the same offsets, each bounded to ±0.2, on
  P1g-b's pools.
- **No GAIN.** The offsets gained on P1c2's pools and not on P1g-b's. Section 1 shows the pools moved the same reads:
  webqsp rose on three of the four lean fits, one of them past the null, and the GNN's musique in domain fell. So
  either the bigger pools already give what the offsets gave, or the offsets' shift does not hold on the pools' added
  rows. That closes this form only. The idea had a signal, so it stays open (docs/SCREENS.md, 3b). A5c's own form for
  no GAIN is A5i: the graph's kind as an input that each model learns how far to trust. On P1g-b's pools it is the
  next form, for both models.
- **INCOMPLETE.** As A5c's.

**The pools.** Which pools the full run uses is P1g-c's call (its section 5). An arm's change joins the full run if it
passed its screen on the pools that run uses:
- **on P1g-b's pools** (P1g-c's pools call ADOPT_FOR_FULL_RUN), this file's verdict decides, per arm;
- **on P1c2's pools** (P1g-c's pools call NOT_ADOPTED), A5c's own verdict on those pools decides (both arms
  ADOPT_FOR_FULL_RUN, `outputs/a5c/screen-verdict.json`), and this file's verdict is reported beside, deciding nothing.

## 6. How it runs

**Self-test.** `python outputs/mp_unified/za5cp.py --selftest` passes on the laptop before this commit (5.2 to 11.1 s
over four runs, 12.3 s with its imports). On toy carves with the KT input and rmatch's chains (typed), and their copy
without chains (untyped), it checks:
- **the arms:** A5c's own objects, on the lean pair's carves (the carves P1g-b's fits read). Each model is its base
  arm's with A5CRet just above zret. The MLP has no message step. webqsp is in no training split. The bases are
  P1g-b's card fits;
- **the roots:** inside, P1g-b's (caches, chains, links, KT columns, gate and declaration): the same roots P1g-c's arms
  read, each one a family P1g-c's gate checks. All are restored after;
- **A5c's model, bit for bit:**
  - from the same seed, each arm has the same state and the same random draws as A5c's;
  - with the offsets and the KT head set, the same scores and every gradient, on a typed batch and on its untyped copy,
    in eval and in training;
  - toy fits through lean_gpu's loop inside P1g-b's roots equal the same fits outside them, bit for bit: every state,
    the SWA state and the forward counts;
  - the offsets move on the typed graph and stay at zero on its untyped copy;
- **the gate,** on toy records of P1g-b's:
  - train and read refuse while it is shut, after a record changed, and when it is another stage's or missing;
  - with it open, a fit refuses a missing or failed P1g-b gate, and KT records that are not P1g-b's;
  - with all in place, the fit and the read run on P1g-b's roots, and the fit is stamped A5cp's (both gates, the
    snapshot, the records it read and its base fit);
- **the refusals:** the lean pair's and P1g-c's arms (zrckp, zspk, zrckpr, zspx, zspxr), another split, another fit
  name (A5c's included), P1c2's caches or none, and the CPU. Read refuses A5c's, P1g-b's and the wrong split's fits;
- **the calls:** relz's pair and re-call per arm against P1g-b's fits, stamped A5cp's. A screen decided against A5c's
  bases, and base screens of A5c's bases, are refused. relz is restored after;
- **the verdict:** ADOPT, the two NOT_ADOPTED outcomes, three INCOMPLETE cases (another stage's re-call, a missing
  one, eleven reads), the next forms, what is beside (with each fit's stamped forwards), and the F1 note.

**Host items** (21, `a5cp-*`):
- **No build.** The fits and reads read P1g-b's caches, KT columns, links and chains, which P1g-c's items rebuild and
  its gate checks. They do not read P1g-b's looks: only builds read those, and P1g-c's last items release them. While
  A5cp's items are queued, the host's clean-up keeps P1g-b's roots: za5cp.py imports zp1gb.py, which names them, and
  the trains and reads name the caches in their command.
- **Base screens:** 4 (`a5cp-base-zrckp`, `-hp`, `a5cp-base-zspk`, `-hp`). Each compares a P1g-b fit with step 1's,
  has no dependency, and takes 1 CPU and 4 GB (as A5c's compares). P1g-c's queued compares (`p1gc-lean-*`) also name
  P1g-b's four fits, so those stay on the host.
- **The screen:** 17 items. Per arm: two trains, two reads, two compares, a pair and a re-call. Then the verdict.
  - **Trains and reads** run on the card at `--frac 0.92`, share 0.94 and 1.1 CPUs, one at a time, as P1g-b's and
    P1g-c's do. The arms read what the base arms read. So memory is P1g-b's measured peak for the base arm's item of
    the same split, × 1.1 + 2 GB. Trains follow P1g-c's gate (`p1gc-gate`). Each read follows its train and the gate.
  - **Compares** (1 CPU, 4 GB) follow their read. **Pairs** (1 CPU, 2 GB) follow both of the arm's compares.
    **Re-calls** (1 CPU, 2 GB) follow the pair and the arm's two base screens. **The verdict** (1 CPU, 2 GB) follows
    both re-calls.
- **Inputs:** P1g-c's train list plus za5c.py, zkind.py and za5cp.py: 48 files, which hold za5cp.py's import
  closure. The re-calls add the four null files, and the verdict adds A5c's two re-calls.
- **Disk:** about 0.3 GB at most. P1c3's fits took about 0.02 GB each with their reads (docs/P1GB_FULL_RUN.md,
  section 4), and P1g-b's pools hold about 3.4 times their rows.
- **ETA:** P1g-b's fits took 972 to 1,739 s each, one at a time on the full card, and a read takes about 7 minutes
  (docs/P1GC_ADOPTED_ON_P1GB.md, section 6). So A5cp's four fits and reads take about 1.6 to 2.4 hours of the card.
  They can start once P1g-c's gate opens. They share the card one at a time with P1g-c's six fits and reads (about 3.5
  hours, the same section), in the order the queue sets. The CPU items take minutes.

**The commands.** One arm and split are shown. zspkc and L-hotpotqa (names ending `-hp`) follow the same pattern. On
the host the trains and reads run under `retry_cmd.py --cuda` and `cuda_alloc.py --frac`, as in the items.

    python outputs/mp_unified/za5cp.py compare --new outputs/screen/fits/scr-p1gb-zrckp \
        --base outputs/step1/fits/L-musique --out outputs/a5cp/base-zrckp-L-musique
    python outputs/mp_unified/za5cp.py train --split L-musique --name scr-a5cp-zrckpc --arm zrckpc --device cuda \
        --host --cache-root outputs/p1gb/cache
    python outputs/mp_unified/za5cp.py read --name scr-a5cp-zrckpc --device cuda --host --cache-root outputs/p1gb/cache
    python outputs/mp_unified/za5cp.py compare --new outputs/screen/fits/scr-a5cp-zrckpc \
        --base outputs/screen/fits/scr-p1gb-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a5cp-zrckpc
    python outputs/mp_unified/za5cp.py pair --arm zrckpc \
        --screens outputs/screen/scr-a5cp-zrckpc.json,outputs/screen/scr-a5cp-zrckpc-hp.json \
        --out outputs/screen/scr-a5cp-zrckpc-pair
    python outputs/mp_unified/za5cp.py recall --arm zrckpc \
        --null outputs/screen/scr-null-s1.json,outputs/screen/scr-null-s2.json,outputs/screen/scr-null-s1-hp.json,outputs/screen/scr-null-s2-hp.json \
        --pair outputs/screen/scr-a5cp-zrckpc-pair.json --out outputs/screen/scr-a5cp-zrckpc-pair-recall
    python outputs/mp_unified/za5cp.py verdict

## 7. Results

Not run yet.
