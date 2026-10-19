# A2xi: prank's ranks in the z-scores' place, on both models

Declared 19 October 2026, before any number. The next form A2h's verdict named for its rank half (zrckpi and zspki:
NOT_ADOPTED with no GAIN; docs/A2H_HALVES_APART.md, section 5, and `outputs/a2h/screen-verdict.json`), under
docs/SCREENS.md section 3b and docs/VERDICT_AUDIT_2026_10_17.md. Code: `outputs/mp_unified/za2xi.py`.

## 1. Why this, now

**prank** (docs/SCREENS.md, "scr-prank (L-musique): MIXED (23:08)") replaced each fixed block's z-score with two rank
columns on step 1's model. Its L-musique fit had the largest zero-shot gain any screen has had, and it paid in domain
(`outputs/screen/scr-prank-pair-recall.md`, under the seed-null re-call):

| read | prank minus step 1, R@5 [95% CI] | floor | re-call |
| --- | --- | ---: | --- |
| L-musique musique (zero-shot) | +0.2016 [+0.1901, +0.2126] | 0.0720 | GAIN |
| L-musique webqsp (zero-shot) | +0.0269 [+0.0133, +0.0410] | 0.0102 | GAIN |
| L-musique metaqa (in domain) | −0.0310 [−0.0351, −0.0269] | 0.0075 | LOSS |
| L-hotpotqa metaqa (in domain) | −0.0320 [−0.0362, −0.0278] | 0.0149 | LOSS |
| L-musique hotpotqa (in domain) | −0.0080 [−0.0108, −0.0054] | 0.0075 | LOSS |
| L-hotpotqa hotpotqa (zero-shot) | −0.0255 [−0.0298, −0.0214] | 0.0200 | LOSS |

The musique gain took step 1's model from 0.270 to 0.471, against rrf's 0.473 (SCREENS.md).

**A2h** (18 October) put the same two columns beside zret's z-scores, on today's lean pair. Both arms were NOT_ADOPTED
with no GAIN. The reads that moved (`outputs/screen/scr-a2h-zrckpi-pair-recall.json` and
`scr-a2h-zspki-pair-recall.json`):

| read | floor | zrckpi (the MLP) | zspki (the GNN) |
| --- | ---: | --- | --- |
| L-musique musique (zero-shot) | 0.0720 | +0.0128 [+0.0080, +0.0178], WITHIN (filed GAIN) | +0.0283 [+0.0221, +0.0346], WITHIN (filed GAIN) |
| L-musique webqsp (zero-shot) | 0.0102 | −0.0142 [−0.0248, −0.0034], LOSS | −0.0137 [−0.0244, −0.0035], LOSS |
| L-hotpotqa webqsp (zero-shot) | 0.0233 | +0.0155 [+0.0046, +0.0261], WITHIN (filed GAIN) | +0.0035 [−0.0056, +0.0126], WITHIN |
| L-musique hotpotqa (in domain) | 0.0075 | −0.0039 [−0.0061, −0.0016], WITHIN | −0.0020 [−0.0041, +0.0003], WITHIN |
| L-hotpotqa 2wiki (in domain) | 0.0075 | −0.0027 [−0.0043, −0.0011], WITHIN | −0.0019 [−0.0035, −0.0003], WITHIN |

metaqa held on both arms (+0.0001 and +0.0010 in the L-musique fits, −0.0015 and −0.0010 in the L-hotpotqa fits, all
WITHIN). Beside the z-scores, the ranks no longer cost what they cost on step 1's model, and they no longer gained.

**Why replacing comes next.** Beside the z-scores, an in-domain fit can lean on the z-scores and leave the ranks (A2h,
section 5). A2xi takes the fixed blocks' z-scores out, so each model must read the ranks. Any loss that brings goes
to the back-off form (section 5).

**What to expect.** The lean pair already reads musique zero-shot above rrf: zrckp 0.5121 and zspk 0.5012, against
rrf's 0.4734 (the rows of A2h's re-calls). prank's +0.202 lifted step 1's model to rrf's level, so it is not the size to
expect here. The lean pair's z-scores are zret's, against the pool's retrieved rows, not step 1's whole-pool ones.
A2xi asks whether the ranks carry what those z-scores do not, when the model must read them.

## 2. The form

Per fixed block b, with w columns and keep k, zret's l1 input `[raw·k, z·k, k]` becomes `[raw·k, hi·k, lo·k, k]`:

| part | definition |
| --- | --- |
| hi | 60 / (60 + min(r, 50)), r the row's competition rank in its column among the question's rows, from the top (0 for the best; tied values share the best position) |
| lo | the same, from the bottom |
| constants | prank's, unchanged (`lean_screen5.rank_inputs`): 60 (rrf's constant), the cap 50 (ztop50's reference size), 16 columns ranked at a time |
| SEMB | zret's `[raw·k, z·k, k]`, unchanged (prank kept SEMB's z-score too) |
| base term | zret's, unchanged: base_w · z(rrf), against the pool's retrieved rows (prank kept step 1's) |
| l1 | 3w + 1 inputs per fixed block in place of 2w + 1. On the card fits (86 fixed columns, SEMB 64): 395 in place of 309, so 128 × 86 = 11,008 more weights |

hi and lo are A2h's rank half, column for column; the selftest checks them bit for bit against `za2h.half_inputs`. A2h
added them beside the z-scores, and A2xi puts them in the z-scores' place. Nothing is built: they are computed in each
batch from the cache's own rows.

**How it enters.**

    pre = l1([raw·k, hi·k, lo·k, k] per fixed block; [raw·k, z·k, k] for SEMB)
    s   = base_w · z(rrf) + out(h)                                   (zret's)

zrckp's KT head and chain match, and zspk's KT mix and propagation, are the base's. The GNN propagates the score A2xi
changes.

**The MLP rule.** hi and lo are computed from the block's own feature values (its raw input columns) over the question's
rows, never from a score. They are made under no_grad from the detached columns, so no gradient leaves them. The
selftest asserts that:
- no weight of the model moves them, so no score enters them;
- another block's values do not move them;
- computed on one question's rows alone, they are the same bit for bit;
- they are prank's ranks by hand, cut at 50 on a 120-row pool.

No score is read, and nothing is normalised over the question's scores. A rank is a within-pool statistic of an input
column, as zret's z-score is.

**The start.**
- l1 is zret's own l1 with the fixed z-score columns taken out and hi's and lo's put in at zero. It is made without
  drawing a random number (`torch.nn.utils.skip_init`); every other weight is the base's, from the same draws.
- out starts at zero, so each arm starts with its base's scores bit for bit, in eval and in training (the dropout masks
  are drawn in the same order).
- The replaced inputs change h, so out's weight is the one shared gradient that differs at the first step.
- A replacing form cannot stay its base once out moves, because the z-scores are gone. With zret's fixed z-score
  weights set to zero in the base, the two scorers agree to float rounding.

**Why hi and lo start at zero.** prank drew a new l1 after the base's draws, which moves every later random number (the
dropout masks). Starting from zret's own draws keeps them, so the arm differs from its base by the replacement alone.

## 3. The arms

| arm | model | carve |
| --- | --- | --- |
| zrckpxi | zrckp's model with each fixed block's z-score replaced by hi and lo (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, A2XMix, ZRet) | zrckp's (`zp1c3.KTChainCarveZRC`) |
| zspkxi | zspk's model with the same replacement (MRO: ZSPK, ZProp, KTMix, ZRM, ChainMatch, A2XMix, ZRet); zsp's propagation reads the score it changes | zspk's (`zp1c3.KTLinkCarveBase`) |

## 4. The fits and the call

A2h's, unchanged, per arm:
- **Fits:** one card fit on L-musique and one on L-hotpotqa (`scr-a2xi-<arm>`, `-hp`), on P1c2's roots with P1c3's KT
  input, at the card fits' settings (variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split (zrckp: scr-p1c3p-zrckp and -hp; zspk:
  scr-p1c3-zspk and -hp), with step 1's fit beside. The re-call's base screens are L1's.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md).

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A2xi's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names each arm's next form (section 5).

**Reported beside, deciding nothing:**
- musique, both splits (the read prank gained on, and Claim 1's read in domain), with the two arms side by side;
- metaqa and hotpotqa (the reads prank lost on step 1's model), and webqsp (A2h's LOSS on both arms);
- hit@1 and FC@5 per read;
- A2h's rank-half deltas (zrckpi, zspki) and prank's, set beside each read when the results are filed.

**Serving cost.** Flagged, to be timed cold before any latency claim. The ranks pay A2h's two sorts per fixed column
per question. The fixed blocks' z-scores are no longer computed; SEMB's and rrf's still are.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools, then its full run, its own declared file |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A2xig, the learned back-off between the ranks and the z-scores: zret's z-score columns of the fixed blocks come back into l1 beside hi and lo (their weights zret's draws, so the scale learns from the first step), times tanh(⟨w, c⟩ + b) per question, w and b zero so the scale starts at 0 (A2xi's form), c the pool's log size and retrieved share (input columns only); on both models, its own declared file |
| NOT_ADOPTED, with no GAIN | A2xix, prank's replacing form on the adopted arms (R1's zrckpr and G5's zspx), on both models, its own declared file; the rank idea closes as a model input only if A2xix finds no GAIN either |
| INCOMPLETE | the missing re-call is run; nothing else changes |

**Why these.**
- **A GAIN with a LOSS** is prank's old trade: musique up, metaqa down. The scale is per pool, and c says what kind of
  pool it is (metaqa's big KB pools against the passage pools). Each model learns where the z-scores come back. c is
  computed from the pool's rows and rrf, so the MLP rule holds.
- **No GAIN:** the ranks have then run beside the z-scores (A2, A2h) and in their place (A2xi) on the lean pair. The
  adopted arms are the models they have not met: R1's zrckpr and G5's zspx, both ADOPT_FOR_FULL_RUN
  (`outputs/r1/screen-verdict.json`, `outputs/g5/screen-verdict.json`). So the idea closes as an input only after
  them (SCREENS.md 3b).

No model change is adopted on today's pools (the user's rule of 13 October): an adopt is refitted on P1g-b's pools
before its full run.

## 6. How it runs

**Self-test.** `python outputs/mp_unified/za2xi.py --selftest` passes before this commit (4.0 s). On rmatch's toy carves
(chains, links), it checks:
- **the arms:** the two MROs, with the replacing mix just above zret's forward; the base arms, A2's and A2h's unchanged;
  no arm is one of theirs; on the card fits' shape l1 takes 395 inputs in place of 309;
- **the columns:** l1's input is zret's with each fixed block's z-score columns replaced by A2h's rank half, bit for
  bit, on the toy's blocks and on three blocks with SEMB between two fixed ones (the third with many ties); the base
  term is zret's; hi and lo are prank's ranks by hand, cut at 50 on a 120-row pool;
- **the MLP rule:** as in section 2;
- **the start:** each arm is its base with no random number drawn, every tensor but l1's weight the base's, and l1's
  kept columns zret's; its scores in eval and training, and every shared gradient but out's weight, are the base's bit
  for bit;
- **neutral and moved:** with zret's fixed z-score weights at zero, the two scorers agree; with hi's and lo's weights
  moved, the scorer is zret's form on A2xi's inputs, and the GNN's propagation reads that score;
- **toy fits** through lean_gpu's loop repeat bit for bit, hi's and lo's weights learn, and a fit loads back through
  lean_gpu's loader;
- **the refusals:** train refuses another arm, split, fit name, cache root or device; read refuses a fit that is not
  A2xi's, not its arm's or not its split's;
- relz's pair and re-call per arm, stamped A2xi's; the verdict's outcomes and next forms, and four kinds of INCOMPLETE
  (another stage, another declaration, another arm, eleven reads) beside a missing re-call.

**Commands**, for zrckpxi (zspkxi's are the same, against scr-p1c3-zspk):

    python outputs/mp_unified/za2xi.py train --split L-musique --name scr-a2xi-zrckpxi --arm zrckpxi --device cuda \
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za2xi.py train --split L-hotpotqa --name scr-a2xi-zrckpxi-hp --arm zrckpxi --device cuda \
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za2xi.py read --name scr-a2xi-zrckpxi --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za2xi.py compare --new outputs/screen/fits/scr-a2xi-zrckpxi \
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a2xi-zrckpxi
    python outputs/mp_unified/za2xi.py pair --arm zrckpxi \
        --screens outputs/screen/scr-a2xi-zrckpxi.json,outputs/screen/scr-a2xi-zrckpxi-hp.json \
        --out outputs/screen/scr-a2xi-zrckpxi-pair
    python outputs/mp_unified/za2xi.py recall --arm zrckpxi \
        --null outputs/screen/scr-null-s1.json,outputs/screen/scr-null-s2.json,outputs/screen/scr-null-s1-hp.json,outputs/screen/scr-null-s2-hp.json \
        --pair outputs/screen/scr-a2xi-zrckpxi-pair.json --out outputs/screen/scr-a2xi-zrckpxi-pair-recall
    python outputs/mp_unified/za2xi.py verdict

On the host each GPU command runs under `outputs/host_ops/retry_cmd.py --cuda` and `outputs/host_ops/cuda_alloc.py`.

**Host items** (17, `a2xi-*`):
- **trains:** 4, with no build to wait on. They run at A2h's fractions (L-musique 0.20, L-hotpotqa 0.27, GPU share
  +0.02) and memory (4.5 GB; 7 for zspk's L-hotpotqa fit). The replacement adds 86 columns to l1 where A2h's rank half
  added 172, with the same sorts;
- **reads:** 4 at 0.27, share 0.29, 6 GB;
- **the call:** 4 compares, 2 pairs, 2 re-calls (after L1's base screens of zrckp and zspk) and the verdict;
- **no build:** the arms read P1c2's caches, P1c3's KT columns and the base's chains and links, as A2h's arms did.

**Disk:** about 0.3 GB of fits and reads.

## 7. Results

Not run yet.
