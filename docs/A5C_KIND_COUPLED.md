# A5c: zck's kind offsets in their coupled form, on both models

Declared 18 October 2026, before any number. The next form A5's verdict named for both arms
(docs/A5_KIND_HEAD_BACKOFF.md, section 7), under docs/SCREENS.md section 3b and docs/VERDICT_AUDIT_2026_10_17.md.
Code: `outputs/mp_unified/za5c.py`. Filed 19 October: both arms ADOPT_FOR_FULL_RUN (section 7).

## 1. Why this, now

**zck** (round 28, 9 October, `zkind.py`) gave zret's output layer and rrf's base weight learned offsets on a batch from
a graph with typed relations (a batch carrying rmatch's chain entries: metaqa and webqsp), trained jointly with the
shared weights. On zrm it gained webqsp zero-shot (+0.0160 and +0.0402) and lost 2wiki in domain (−0.0082), on batches
the offsets never act on, through the shared weights.

**A5** (17 October) took that path away: the head over zret's *detached* hidden units, its gradient kept off the shared
weights. Both arms NOT_ADOPTED with nothing past the null: the passage graphs moved by exactly 0 (the shared weights
were the base's), and webqsp zero-shot rose within its floor (MLP +0.0176, GNN +0.0136 on L-hotpotqa).

So the form that carried zck's gain, the coupled one, never ran on the lean pair. A5's section 5 fixed it as the next
form for both arms.

## 2. The form

zck as filed (`zkind.ZKind`'s forward), inside each of today's models:

- **On a typed batch**, zret's score is (base_w + kd_base) · z(rrf) + (W + kd_w) · h + (b + kd_b), where W and b are
  zret's output layer, h its last hidden units and z(rrf) its base term. The three offsets start at zero.
- **Any other batch** is zret's forward, unchanged.
- **The loss is the model's own**, so the offsets and the shared weights train jointly (zck's coupling).

The rest of each model reads zret's score as before:

| model | what reads the offset score |
| --- | --- |
| zrckp (MLP) | rmatch's chain term and the pointwise KT head add to it |
| zspk (GNN) | the KT head reads it, and zsp's propagation reads the sum, so the offsets act before the step |

**The graph's kind** is the graph's own property: a batch carries chain entries exactly when its graph has typed
relations. On today's six that is metaqa and webqsp; webqsp never trains, so in training only metaqa's batches move the
offsets.

**The MLP rule.** No score read, nothing normalised over the question's scores; the offsets are weights.

**Serving cost.** One added product per row on a typed graph (zck's). The passage graphs serve as the base does.

## 3. The arms

| arm | model | carve |
| --- | --- | --- |
| zrckpc | zrckp's model with A5CRet just above zret (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, A5CRet, ZRet) | zrckp's (`zp1c3.KTChainCarveZRC`) |
| zspkc | zspk's model with A5CRet just above zret (MRO: ZSPK, ZProp, KTMix, ZRM, ChainMatch, A5CRet, ZRet) | zspk's (`zp1c3.KTLinkCarveBase`) |

## 4. The fits and the call

A5's, unchanged:
- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-a5c-<arm>`, `-hp`), on P1c2's roots with
  P1c3's KT input, at the card fits' settings (variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128). Both splits
  train on metaqa, so both fits move the offsets.
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split (zrckp: scr-p1c3p-zrckp and -hp; zspk:
  scr-p1c3-zspk and -hp), step 1's fit beside. The re-call's base screens are L1's.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), per arm.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A5c's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:** the typed graphs (metaqa in domain, webqsp zero-shot: zck's gain); 2wiki in
domain (zck's loss, its known risk); musique in domain (Claim 1's read), both arms side by side; hit@1 and FC@5 per
read; per fit, the typed and untyped training forwards.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools, then its full run |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A5cb, the coupled offsets bounded (each offset 0.2 tanh of its parameter), on both models |
| NOT_ADOPTED, with no GAIN | A5i, the graph's kind as a 0/1 input through R1's pointwise head (the learned back-off), on both models |
| INCOMPLETE | the missing re-call is run; nothing else changes |

No model change is adopted on today's pools (the user's rule of 13 October): an adopt is refitted on P1g-b's pools
before its full run.

## 6. How it runs

**Self-test.** `python outputs/mp_unified/za5c.py --selftest` passes before this commit (9.9 s). On toy carves with the
KT input and rmatch's chains (typed) and their copy without chains (untyped, zkind's `Untyped`), it checks:
- the arms' MROs with A5CRet just above zret; the base arms unchanged and A5's arms not imported; zck's offsets' names;
- from the same seed, each arm is its base's state plus three zero offsets, with no draw;
- at zero, each arm is its base bit for bit, typed or not, in eval and in training: scores and every shared gradient.
  On a typed batch each offset's gradient equals its weight's (the coupling); on an untyped batch the offsets get none;
- with the offsets set, a typed batch equals the base model with its output layer and base weight moved by them, bit
  for bit (scores and every gradient), and an untyped batch is the base bit for bit;
- toy fits through lean_gpu's loop repeat bit for bit. On the typed graph the offsets move and the shared weights leave
  the base fit's; on its untyped copy every state is the base fit's bit for bit and the offsets stay at zero;
- the refusals; relz's pair and re-call per arm, stamped A5c's; the verdict's outcomes and next forms.

**Host items** (17, `a5c-*`) go right after A4f's, sized as A5's (three parameters more than the base): trains and
reads at A5's `--frac` and GPU share, memory as A5's items.

## 7. Results

Filed 19 October 2026 from the host's records (`outputs/a5c/screen-verdict.json`, written 2026-10-18T14:59:12Z; the
re-calls `outputs/screen/scr-a5c-zrckpc-pair-recall.json` and `-zspkc-`, host paths made relative). **Both arms
ADOPT_FOR_FULL_RUN:** each has two GAINs and no LOSS after the seed-null re-call.

- **zrckpc: 2 GAIN, 0 LOSS.** webqsp zero-shot rises on both splits: +0.0189 [+0.0089, +0.0294] on L-musique and +0.0239
  [+0.0122, +0.0359] on L-hotpotqa. musique zero-shot rises +0.0084 (filed GAIN, re-called WITHIN against its 0.072
  floor); metaqa in domain -0.0025 on L-musique, WITHIN.
- **zspkc: 2 GAIN, 0 LOSS.** webqsp zero-shot rises +0.0520 [+0.0392, +0.0648] on L-hotpotqa, the largest rise of the
  thirteen screens filed today, and musique in domain (L-hotpotqa, Claim 1's read) rises +0.0083 [+0.0036, +0.0131].
  musique zero-shot rises +0.0224 (filed GAIN, re-called WITHIN); 2wiki in domain +0.0041 on L-hotpotqa and webqsp
  zero-shot -0.0052 on L-musique, both WITHIN.
- **Beside:** metaqa in domain, the typed graph the offsets train on, moves -0.0025 and -0.0005 (MLP), +0.0021 and
  +0.0006 (GNN); 2wiki in domain +0.0001 / -0.0018 (MLP) and -0.0014 / +0.0041 (GNN); musique in domain -0.0018 on the
  MLP. A5, the kind head with its gradient kept off the shared weights, had moved webqsp zero-shot +0.0176 and +0.0136
  on L-hotpotqa (WITHIN); with the offsets coupled to the shared weights the rise passes the seed null (the MLP on both
  splits, the GNN on L-hotpotqa).
- **The fits:** a quarter of the training forwards are typed (table below).

This adopts the form for its full run only. The next form, fixed in section 5 and recorded in the verdict, is each arm
fitted again on P1g-b's pools, then its full run, in its own declared file. Both arms passed, so if the refits and full
runs hold, the coupled kind offsets join both lean models, and F1's rule (one input set for both) holds.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's; a call is shown with its
filed call where the re-call changed it; base R@5 shown as zrckp's / zspk's):

| split | dataset | read | base R@5 | zrckpc: delta [95% CI], call | zspkc: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8019 / 0.8009 | -0.0025 [-0.0036, -0.0016], WITHIN | +0.0021 [+0.0011, +0.0031], WITHIN |
| L-musique | squad | in-domain | 0.9116 / 0.9114 | -0.0018 [-0.0035, +0.0000], WITHIN | -0.0012 [-0.0031, +0.0007], WITHIN |
| L-musique | musique | zero-shot | 0.5121 / 0.5012 | +0.0084 [+0.0034, +0.0135], WITHIN (filed GAIN) | +0.0224 [+0.0162, +0.0290], WITHIN (filed GAIN) |
| L-musique | hotpotqa | in-domain | 0.9065 / 0.9080 | -0.0001 [-0.0020, +0.0020], WITHIN | -0.0016 [-0.0035, +0.0005], WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 / 0.9383 | +0.0001 [-0.0014, +0.0016], WITHIN | -0.0014 [-0.0027, -0.0001], WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 / 0.3229 | +0.0189 [+0.0089, +0.0294], **GAIN** | -0.0052 [-0.0174, +0.0061], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8044 / 0.8040 | -0.0005 [-0.0014, +0.0005], WITHIN | +0.0006 [-0.0004, +0.0016], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 / 0.9135 | -0.0006 [-0.0024, +0.0013], WITHIN | -0.0014 [-0.0032, +0.0003], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 / 0.5754 | -0.0018 [-0.0066, +0.0030], WITHIN | +0.0083 [+0.0036, +0.0131], **GAIN** |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 / 0.8499 | -0.0036 [-0.0064, -0.0008], WITHIN | -0.0031 [-0.0061, -0.0003], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 / 0.9333 | -0.0018 [-0.0032, -0.0002], WITHIN | +0.0041 [+0.0026, +0.0056], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 / 0.2680 | +0.0239 [+0.0122, +0.0359], **GAIN** | +0.0520 [+0.0392, +0.0648], **GAIN** |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a5c-zrckpc.json` and the others, `-hp` for L-hotpotqa); and zrckpk's R@5 delta (A5, its own
re-call) and zspkk's R@5 delta (A5, its own re-call), on the same read:

| split | dataset | zrckpc FC@5 | zrckpc hit@1 | zspkc FC@5 | zspkc hit@1 | zrckpk R@5 (A5) | zspkk R@5 (A5) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L-musique | metaqa | -0.0022 | -0.0141 | +0.0025 | +0.0089 | -0.0019 | -0.0008 |
| L-musique | squad | -0.0018 | -0.0003 | -0.0012 | -0.0001 | +0.0000 | +0.0000 |
| L-musique | musique | +0.0116 | -0.0348 | +0.0302 | -0.0410 | +0.0000 | +0.0000 |
| L-musique | hotpotqa | +0.0011 | -0.0076 | -0.0035 | +0.0019 | +0.0000 | +0.0000 |
| L-musique | 2wiki | +0.0023 | -0.0022 | -0.0013 | -0.0070 | +0.0000 | +0.0000 |
| L-musique | webqsp | +0.0120 | +0.0080 | -0.0053 | +0.0047 | -0.0005 | +0.0066 |
| L-hotpotqa | metaqa | -0.0003 | -0.0021 | +0.0015 | -0.0020 | -0.0005 | -0.0003 |
| L-hotpotqa | squad | -0.0006 | -0.0014 | -0.0014 | +0.0011 | +0.0000 | +0.0000 |
| L-hotpotqa | musique | -0.0012 | -0.0033 | +0.0037 | -0.0095 | +0.0000 | +0.0000 |
| L-hotpotqa | hotpotqa | -0.0076 | -0.0115 | -0.0038 | +0.0024 | +0.0000 | +0.0000 |
| L-hotpotqa | 2wiki | -0.0052 | +0.0004 | +0.0116 | -0.0016 | +0.0000 | +0.0000 |
| L-hotpotqa | webqsp | +0.0160 | +0.0200 | +0.0426 | +0.0306 | +0.0176 | +0.0136 |

The fits (stamped in each fit's screen.json; a typed forward is a batch of metaqa's questions, where the kind offsets
act):

| fit | typed forwards (metaqa's) | untyped forwards | typed share |
| --- | ---: | ---: | ---: |
| L-musique | 5,918 | 17,759 | 0.250 |
| L-hotpotqa | 5,591 | 16,771 | 0.250 |
