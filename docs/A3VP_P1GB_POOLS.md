# A3vp: A3v's two-sided share rule on the MLP, fitted again on P1g-b's pools

Declared 19 October 2026, before any number. The next form A3v's verdict named for its MLP arm
(docs/A3V_VSHARE_TWO_SIDED.md, sections 5 and 7; `outputs/a3v/screen-verdict.json`), under the user's rule of 13
October that no model change is adopted on today's pools. Code: `outputs/mp_unified/za3vp.py`. Numbers here are
development numbers.

## 1. Why this, now

**A3v** (19 October) trained the lean pair with vshare's two-sided share rule in zret's reference: in training only, a
pool's retrieved rows hold a drawn share t of the reference, with t from A3's rich range on a pool whose own retrieved
share is under 0.375 and from A3t's thin range on any other. Its MLP arm, zrckpy, was ADOPT_FOR_FULL_RUN on P1c2's
pools, with one GAIN and no LOSS after the seed-null re-call. Its GNN arm, zspky, was NOT_ADOPTED with no GAIN
(`outputs/a3v/screen-verdict.json`, 2026-10-19T13:44:28Z).

The MLP's GAIN, from its re-call (`outputs/screen/scr-a3v-zrckpy-pair-recall.json`; delta = the arm's R@5 minus its
base's, with its 95% CI):

| arm | split | dataset | read | base, arm | delta [95% CI] | floor |
| --- | --- | --- | --- | --- | --- | ---: |
| zrckpy | L-musique | webqsp | zero-shot | 0.3627, 0.3740 | +0.0112 [+0.0008, +0.0211] | 0.0102 |

musique zero-shot on L-musique fell -0.0111 [-0.0167, -0.0054]: filed LOSS, re-called WITHIN against its 0.0720
floor.

**The rule of 13 October.** No model change is adopted on today's pools. So A3v's verdict named the MLP arm's next
form: the arm fitted again on P1g-b's pools, then its full run, each in its own declared file. This is that refit. The
GNN arm's next form is A3z, the share idea as an input on both models. That is not a refit, so A3vp has one arm.

**P1g-b's pools** are P1g2's chosen pools: ALL ≥ 0.95 on every dataset, with about 3.4 times today's rows
(docs/P1GC_ADOPTED_ON_P1GB.md, section 1). P1g-b refit the lean pair on them, unchanged, and was NOT_ADOPTED
(`outputs/p1gb/screen-verdict.json`, 2026-10-17T17:41:13Z). P1g-c (declared 18 October) puts them under the adopted
models, and its verdict decides the pools.

**The pools alone moved the same reads, and as far.** On the reads where A3v moved the MLP, P1g-b's refit moved zrckp
(`outputs/screen/scr-p1gb-zrckp-pair-recall.json`; delta = zrckp on P1g-b's pools minus zrckp on P1c2's):

| read | A3v's rule, on P1c2's pools | the pools alone, on zrckp | zrckp on P1g-b's pools (A3vp's base R@5) |
| --- | --- | --- | ---: |
| L-musique webqsp, zero-shot (A3v's GAIN) | +0.0112, GAIN | +0.0141 [+0.0015, +0.0257], GAIN | 0.3768 |
| L-musique musique, zero-shot | -0.0111, WITHIN (filed LOSS) | -0.0108 [-0.0171, -0.0048], WITHIN | 0.5013 |
| L-hotpotqa musique, in domain (Claim 1's read) | +0.0004, WITHIN | -0.0050 [-0.0103, +0.0002], WITHIN | 0.5695 |

So on P1g-b's pools zrckp already starts where A3v's rule took it on the GAIN read, and a little past. Only a refit
there says whether the rule still adds anything.

**The pools also change the rule's input.** A3v picks each pool's range from the pool's own retrieved share
r / (r + u). The pools change, so that share changes, and so may the split between the two sides. On P1c2's pools,
A3v's fits reweighted 71,020 and 65,875 question visits, of which 22,776 and 22,711 took the rich range
(docs/A3V_VSHARE_TWO_SIDED.md, section 7). Each A3vp fit stamps its own counts, and the verdict reports them.

## 2. The form

A3v's MLP arm, unchanged: `za3v.ZRCKPY` on its own carve, with A3v's settings (docs/A3V_VSHARE_TWO_SIDED.md, section
2). Only the pools change.

- **The rule.** In a training forward only, each question with at least two retrieved rows (rrf > 0) and at least one
  other row is drawn with probability 0.5 (A3's draws). For a drawn question, the reference that zret takes its
  z-scores against gives the retrieved rows a share t. t ~ U[0.55, 0.95] when the pool's own retrieved share is under
  0.375 (A3's rich range); t ~ U[0.08, 0.20] otherwise (A3t's thin range). Reads are never reweighted: t = 1, zret's
  reference. There is no new parameter.
- **The roots.** The arm reads P1g-b's caches, KT columns and zrc's chains, as P1g-c's arms do. It runs inside
  `zp1gb.on_p1gb()`, which sets the same roots that P1g-c's own context sets for every carve the arm reads.
- **No build.** The arm's carve is zrckp's own (`zp1c3.KTChainCarveZRC`): the one P1g-b's fit of zrckp reads. The rule
  reads only the retrieved flag (rrf > 0) that zret's reference already uses. zrc's chains and the KT columns are
  records P1g-c rebuilds and its gate checks (`zp1gc.py`, RECS). So A3vp needs nothing beyond what zrckp reads, and it
  builds nothing.
- **The gate.** Every fit and read checks P1g-c's gate first (`zp1gc.gate_open`), as zp1gc.py does. Each refuses while
  the gate is shut, missing or another stage's, and when a P1g-b record changed after the gate opened. A fit also checks
  P1g-b's own gate (`zp1gb.gate_passed`). Fits and reads check that the KT records they read are P1g-b's.
- **The MLP rule.** zrckpy reads no score and normalises nothing over the question's scores. The side comes from the
  count of the pool's retrieved rows. The rule changes, in training only, the reference of within-pool statistics of
  input columns. It passes no message.
- **Serving cost.** None. Reads are zret's forward, so the served model is zrckp's form with its own weights. What the
  bigger pools cost is P1g-c's question, not this file's.

## 3. The arm

| arm | model | carve | its base, on P1g-b's pools |
| --- | --- | --- | --- |
| zrckpy | A3v's MLP: zrckp's model with A3VMix just above A3Mix (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, A3VMix, A3Mix, ZRet) | zrckp's (`zp1c3.KTChainCarveZRC`) | P1g-b's `scr-p1gb-zrckp`, `-hp` |

## 4. The fits and the call

A3v's, on P1g-b's pools:
- **Fits:** one card fit on L-musique and one on L-hotpotqa (`scr-a3vp-zrckpy`, `-hp`). They run on P1g-b's roots with
  P1g-b's KT columns, at the card fits' settings (variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128).
- **Reads:** each fit on the six s1eval carves of P1g-b's pools at p@swa.
- **Compares:** each fit against P1g-b's card fit of zrckp on the same split (scr-p1gb-zrckp and -hp), with step 1's
  fit beside. Both fits read the same pools, questions and gold totals.
- **The re-call's base screens:** P1g-b's two zrckp fits against step 1's (`outputs/a3vp/base-zrckp-<split>.json`).
  They are this stage's own. P1g-b's compares put its bases on P1c2's pools first, and the re-call needs step 1's
  first.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md). GAIN, LOSS and
  WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

A3v's rule:
- **ADOPT_FOR_FULL_RUN:** the arm's re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A3vp's re-call of the arm, or does not hold the twelve reads.

The verdict closes only this form (docs/SCREENS.md, section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- metaqa, musique, squad and webqsp on both splits (A3v's beside set). webqsp zero-shot held A3v's GAIN, musique
  zero-shot its filed LOSS, and metaqa vshare's one LOSS. squad's pools are never reweighted;
- musique in domain (L-hotpotqa, Claim 1's read);
- hit@1 and FC@5 per read, from the compares;
- each read's delta beside A3v's delta and call on the same read on P1c2's pools (from A3v's re-call);
- per fit, the reweighting counts it stamped: the question visits, those eligible, and those reweighted on each side.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | A3v's two-sided share rule joins the full run's MLP. That full run is one file, declared before its numbers, after P1g-c's, A5cp's, A7p's and A3vp's verdicts, with the composition P1g-c's section 5 sets. The GNN's share form is A3z's to decide |
| NOT_ADOPTED, with a GAIN and a LOSS | **A3vs** on P1g-b's pools: A3v's rule with the reference's retrieved share t as an input column (1 at every read), the back-off form A3v's section 5 names, its own declared file |
| NOT_ADOPTED, with no GAIN | **A3z's inputs** on P1g-b's pools: each block's and rrf's whole-pool z-score beside zret's, through zero-started columns, its own declared file. The idea stays open |
| INCOMPLETE | the missing re-call is run; nothing else changes |

**Why these.**
- **ADOPT.** The rule then passed on both pools. It joins the full run's MLP as the other adopted changes do, in the
  composition P1g-c's section 5 sets (its pools, its MLP and its GNN). The rule changes the MLP's training, not its
  inputs, so F1's one input set is unchanged. The full run's file names the training rule as a difference between its
  two models, unless A3z settles the GNN's share form first. The verdict records this.
- **A GAIN with a LOSS.** A3v's own form for this outcome (its section 5): the same rule, with t as an input column
  so the model learns what the reweighting changes, on P1g-b's pools.
- **No GAIN.** The rule gained on P1c2's pools and not on P1g-b's. Section 1 shows the pools alone moved the GAIN read
  as far. So either the bigger pools already give what the rule gave, or the rule's shift does not hold on the pools'
  added rows. That closes this form only. The idea had a signal on the MLP, so it stays open (docs/SCREENS.md, 3b).
  A3v's own form for no GAIN is A3z: the share as an input that the model learns how far to trust. On P1g-b's pools it
  is the next form.
- **INCOMPLETE.** As A3v's.

**The pools.** Which pools the full run uses is P1g-c's call (its section 5). The rule joins the full run's MLP if it
passed its screen on the pools that run uses:
- **on P1g-b's pools** (P1g-c's pools call ADOPT_FOR_FULL_RUN), this file's verdict decides;
- **on P1c2's pools** (P1g-c's pools call NOT_ADOPTED), A3v's own verdict on those pools decides (zrckpy
  ADOPT_FOR_FULL_RUN, `outputs/a3v/screen-verdict.json`), and this file's verdict is reported beside, deciding
  nothing.

## 6. How it runs

**Self-test.** `python outputs/mp_unified/za3vp.py --selftest` passes on the laptop before this commit (7.2 s, run
from the repo root, as on the host: the cache-root check resolves against it). On A3v's toy carve, with the KT input
and zrc's chains, it checks:
- **the arm:** A3v's own object, on the lean MLP's carve (the carve P1g-b's fit of zrckp reads). Its model is P1g-b's
  zrckp with A3VMix and A3Mix just above zret, with no message step. webqsp is in no training split. The base is
  P1g-b's card fit. A3v's GNN arm is not here;
- **the roots:** inside, P1g-b's (caches, chains, links, KT columns, gate and declaration): the same roots P1g-c's arms
  read, each one a family P1g-c's gate checks. All are restored after;
- **A3v's model, bit for bit:**
  - from the same seed, the arm has the same state and the same share draws (A3's generator) as A3v's;
  - with zret's output layer and the KT head set, the same scores and every gradient in eval and in training, on a
    batch whose eligible pools sit on both sides of 0.375;
  - toy fits through lean_gpu's loop inside P1g-b's roots equal the same fits outside them, bit for bit: every state,
    the SWA state and A3v's counts, with questions reweighted on both sides;
- **the gate,** on toy records of P1g-b's:
  - train and read refuse while it is shut, after a record changed, and when it is another stage's or missing;
  - with it open, a fit refuses a missing or failed P1g-b gate, and KT records that are not P1g-b's;
  - with all in place, the fit and the read run on P1g-b's roots, and the fit is stamped A3vp's (both gates, the
    snapshot, the records it read, its base fit, A3v's settings and the reweighting counts);
- **the refusals:** A3v's GNN arm, the lean pair's, A3's, A3t's, A5c's and P1g-c's arms (zspky, zrckp, zspk, zrckpv,
  zrckpu, zrckpc, zrckpr, zspx, zspxr), another split, another fit name (A3v's included), P1c2's caches or none, and the
  CPU. Read refuses A3v's, P1g-b's and the wrong split's fits;
- **the calls:** relz's pair and re-call against P1g-b's fits, stamped A3vp's. A screen decided against A3v's base, and
  base screens of A3v's base, are refused. relz is restored after;
- **the verdict:** ADOPT, the two NOT_ADOPTED outcomes, five INCOMPLETE cases (another stage's re-call, another
  declaration's, another arm's, a missing one, and eleven reads), the next forms, what is beside (with each fit's
  stamped counts), and the F1 note.

**Host items** (11, `a3vp-*`):
- **No build.** The fits and reads read P1g-b's caches, KT columns and zrc's chains, which P1g-c's items rebuild and
  its gate checks. They do not read P1g-b's looks: only builds read those, and P1g-c's last items release them. While
  A3vp's items are queued, the host's clean-up keeps P1g-b's roots: za3vp.py imports zp1gb.py, which names them, and
  the trains and reads name the caches in their command.
- **Base screens:** 2 (`a3vp-base-zrckp`, `-hp`). Each compares a P1g-b fit with step 1's, has no dependency, and takes
  1 CPU and 4 GB (as A3v's compares).
- **The screen:** 9 items: two trains, two reads, two compares, a pair, a re-call and the verdict.
  - **Trains and reads** run on the card at `--frac 0.92`, share 0.94 and 1.1 CPUs, one at a time, as P1g-b's, P1g-c's
    and A5cp's do. The arm reads what zrckp reads. So memory is P1g-b's measured peak for zrckp's item of the same
    split, × 1.1 + 2 GB. Trains follow P1g-c's gate (`p1gc-gate`). Each read follows its train and the gate.
  - **Compares** (1 CPU, 4 GB) follow their read. **The pair** (1 CPU, 2 GB) follows both compares. **The re-call**
    (1 CPU, 2 GB) follows the pair and the two base screens. **The verdict** (1 CPU, 2 GB) follows the re-call.
- **Inputs:** P1g-c's train list plus A3v's modules (za3vshare.py, lean_screen7.py, za3t.py, za3v.py) and za3vp.py: 50
  files, which hold za3vp.py's import closure. The re-call adds the four null files, and the verdict adds A3v's
  re-call of zrckpy.
- **Disk:** about 0.15 GB at most. P1c3's fits took about 0.02 GB each with their reads (docs/P1GB_FULL_RUN.md,
  section 4), and P1g-b's pools hold about 3.4 times their rows.
- **ETA:** P1g-b's fits took 972 to 1,739 s each, one at a time on the full card, and a read takes about 7 minutes
  (docs/P1GC_ADOPTED_ON_P1GB.md, section 6). So A3vp's two fits and reads take about 0.8 to 1.2 hours of the card. They
  can start once P1g-c's gate opens, and they queue behind P1g-c's six fits and reads (about 3.5 hours, the same
  section) and A5cp's four. The CPU items take minutes.

**The commands.** L-musique is shown. L-hotpotqa (names ending `-hp`) follows the same pattern. On the host the trains
and reads run under `retry_cmd.py --cuda` and `cuda_alloc.py --frac`, as in the items.

    python outputs/mp_unified/za3vp.py compare --new outputs/screen/fits/scr-p1gb-zrckp \
        --base outputs/step1/fits/L-musique --out outputs/a3vp/base-zrckp-L-musique
    python outputs/mp_unified/za3vp.py train --split L-musique --name scr-a3vp-zrckpy --arm zrckpy --device cuda \
        --host --cache-root outputs/p1gb/cache
    python outputs/mp_unified/za3vp.py read --name scr-a3vp-zrckpy --device cuda --host --cache-root outputs/p1gb/cache
    python outputs/mp_unified/za3vp.py compare --new outputs/screen/fits/scr-a3vp-zrckpy \
        --base outputs/screen/fits/scr-p1gb-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a3vp-zrckpy
    python outputs/mp_unified/za3vp.py pair --arm zrckpy \
        --screens outputs/screen/scr-a3vp-zrckpy.json,outputs/screen/scr-a3vp-zrckpy-hp.json \
        --out outputs/screen/scr-a3vp-zrckpy-pair
    python outputs/mp_unified/za3vp.py recall --arm zrckpy \
        --null outputs/screen/scr-null-s1.json,outputs/screen/scr-null-s2.json,outputs/screen/scr-null-s1-hp.json,outputs/screen/scr-null-s2-hp.json \
        --pair outputs/screen/scr-a3vp-zrckpy-pair.json --out outputs/screen/scr-a3vp-zrckpy-pair-recall
    python outputs/mp_unified/za3vp.py verdict

## 7. Results

Filed 27 October 2026 from the host's records (`outputs/a3vp/screen-verdict.json`, written 2026-10-20T04:49:51Z; the
re-call `outputs/screen/scr-a3vp-zrckpy-pair-recall.json`, host paths made relative). **On P1g-b's pools, NOT_ADOPTED,
with no GAIN and one LOSS. Beside, deciding nothing:** P1g-c's pools call is NOT_ADOPTED (docs/P1GC_ADOPTED_ON_P1GB.md,
section 8), so by section 5's pools rule A3v's own verdict on P1c2's pools decides (zrckpy ADOPT_FOR_FULL_RUN,
`outputs/a3v/screen-verdict.json`).

A3vp ran on 20 October, before N2's gate `n2-gate-a3v` (docs/N2_FRESH_SEED_NULL.md, sections 5 and 6, a feeder edit) was
among its trains' dependencies. N2 decides only whether the refits run, not their calls; its own results are not filed
yet.

- **zrckpy: 0 GAIN, 1 LOSS.** The LOSS is webqsp zero-shot on L-musique: -0.0113 [-0.0206, -0.0014] against its 0.0102
  floor, with FC@5 -0.0173. webqsp zero-shot on L-hotpotqa falls -0.0181 [-0.0295, -0.0069], a LOSS by its pair's
  interval that is WITHIN under its 0.0233 floor (FC@5 -0.0180). Seven more intervals exclude 0, all inside their
  floors: on L-musique, metaqa +0.0012 [+0.0001, +0.0022], musique zero-shot +0.0054 [+0.0010, +0.0098] and 2wiki
  -0.0069 [-0.0084, -0.0054]; on L-hotpotqa, metaqa +0.0030 [+0.0021, +0.0039], musique -0.0058 [-0.0114, -0.0003],
  hotpotqa zero-shot -0.0037 [-0.0067, -0.0007] and 2wiki -0.0054 [-0.0071, -0.0036].
- **Beside: the rule's GAIN on P1c2's pools turns to a LOSS on P1g-b's.** On P1c2's pools (A3v), zrckpy gained webqsp
  zero-shot on L-musique (+0.0112); on P1g-b's pools that read falls -0.0113.

The next form, fixed in section 5 and recorded in the verdict for no GAIN, is that the share idea stays open on the MLP
(it gained on P1c2's pools; docs/SCREENS.md, 3b): A3z's whole-pool z-scores beside zret's on P1g-b's pools, its own
declared file. The verdict's F1 note: zrckpy not adopted: the full run's MLP trains as zrckp does. Both name P1g-b's
pools, which P1g-c has since left out of the full run; on P1c2's pools, where it runs, A3v's verdict adopts zrckpy. A3z,
the whole-pool z-scores on P1c2's pools against the lean pair, has no GAIN on either model (docs/A3Z_WHOLE_POOL_Z.md,
section 7; its next form is A3zx).

The floors are the seed null of 7 October (docs/SCREENS.md section 2: `outputs/screen/scr-null-s1.json`,
`scr-null-s2.json` and their `-hp` files), not N2's fresh floors (docs/N2_FRESH_SEED_NULL.md).

Records, filed with this section: the verdict `outputs/a3vp/screen-verdict.json` and `.md`; for the arm `zrckpy`, the
compares `outputs/screen/scr-a3vp-zrckpy.json` and `-hp.json`, the pair `-pair.json` and its re-call
`-pair-recall.json`, each with its `.md`; the re-calls' base screens `outputs/a3vp/base-zrckp-L-hotpotqa.json`,
`outputs/a3vp/base-zrckp-L-musique.json`, each with its `.md`. The host paths were made relative in each re-call's two
`rel` entries and nowhere else, so the `recall_sha256` each verdict holds is the host file's, from before that change.

The twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's, zrckp's card fit of the split on
P1g-b's pools, P1g-b's `scr-p1gb-zrckp`; a call is shown with its filed call where the re-call changed it):

| split | dataset | read | zrckp R@5 | zrckpy: delta [95% CI], call |
| --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.7993 | +0.0012 [+0.0001, +0.0022], WITHIN |
| L-musique | squad | in-domain | 0.9127 | -0.0010 [-0.0028, +0.0008], WITHIN |
| L-musique | musique | zero-shot | 0.5013 | +0.0054 [+0.0010, +0.0098], WITHIN |
| L-musique | hotpotqa | in-domain | 0.9035 | +0.0007 [-0.0016, +0.0030], WITHIN |
| L-musique | 2wiki | in-domain | 0.9377 | -0.0069 [-0.0084, -0.0054], WITHIN |
| L-musique | webqsp | zero-shot | 0.3768 | -0.0113 [-0.0206, -0.0014], **LOSS** |
| L-hotpotqa | metaqa | in-domain | 0.7997 | +0.0030 [+0.0021, +0.0039], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9122 | +0.0003 [-0.0016, +0.0022], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5695 | -0.0058 [-0.0114, -0.0003], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8546 | -0.0037 [-0.0067, -0.0007], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9340 | -0.0054 [-0.0071, -0.0036], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3364 | -0.0181 [-0.0295, -0.0069], WITHIN (filed LOSS) |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a3vp-zrckpy.json` and `-hp.json`); and A3v's R@5 deltas with the same arm on P1c2's pools, against
zrckp there (its own re-call, `outputs/screen/scr-a3v-zrckpy-pair-recall.json`), on the same read (a call is shown where
it is not WITHIN):

| split | dataset | zrckpy FC@5 | zrckpy hit@1 | zrckpy on P1c2's pools (A3v) R@5 |
| --- | --- | ---: | ---: | ---: |
| L-musique | metaqa | +0.0015 | +0.0119 | +0.0003 |
| L-musique | squad | -0.0010 | -0.0008 | +0.0015 |
| L-musique | musique | +0.0054 | +0.0062 | -0.0111 |
| L-musique | hotpotqa | -0.0007 | +0.0065 | -0.0022 |
| L-musique | 2wiki | -0.0192 | +0.0051 | +0.0014 |
| L-musique | webqsp | -0.0173 | -0.0140 | +0.0112 GAIN |
| L-hotpotqa | metaqa | +0.0038 | +0.0106 | -0.0019 |
| L-hotpotqa | squad | +0.0003 | +0.0013 | +0.0003 |
| L-hotpotqa | musique | -0.0062 | -0.0128 | +0.0004 |
| L-hotpotqa | hotpotqa | -0.0070 | -0.0127 | +0.0038 |
| L-hotpotqa | 2wiki | -0.0118 | +0.0053 | -0.0013 |
| L-hotpotqa | webqsp | -0.0180 | -0.0106 | +0.0036 |
